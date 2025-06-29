/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use derive_more::Display;
use hyperactor::ActorRef;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Named;
use hyperactor::RefClient;
use hyperactor::data::Serialized;
use hyperactor::reference::ActorId;
use pyo3::FromPyObject;
use pyo3::IntoPyObject;
use pyo3::IntoPyObjectExt;
use pyo3::types::PyAnyMethods;
use serde::Deserialize;
use serde::Serialize;

use crate::client::ClientActor;
use crate::debugger::DebuggerAction;
use crate::worker::Ref;

/// Used to represent a slice of ranks. This is used to send messages to a subset of workers.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Ranks {
    Slice(ndslice::Slice),
    SliceList(Vec<ndslice::Slice>),
}

impl Ranks {
    pub fn iter_slices<'a>(&'a self) -> std::slice::Iter<'a, ndslice::Slice> {
        match self {
            Self::Slice(slice) => std::slice::from_ref(slice).iter(),
            Self::SliceList(slices) => slices.iter(),
        }
    }
}

/// The sequence number of the operation (message sent to a set of workers). Sequence numbers are
/// generated by the client, and are strictly increasing.
#[derive(
    Debug,
    Serialize,
    Deserialize,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Hash,
    Ord,
    Copy,
    Named
)]
pub struct Seq(u64);

impl Seq {
    /// Returns the next logical sequence number.
    #[inline]
    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }

    pub fn iter_between(start: Self, end: Self) -> impl Iterator<Item = Self> {
        (start.0..end.0).map(Self)
    }
}

impl Default for Seq {
    #[inline]
    fn default() -> Self {
        Self(0)
    }
}

impl Display for Seq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "s{}", self.0)
    }
}

impl From<u64> for Seq {
    #[inline]
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<Seq> for u64 {
    #[inline]
    fn from(value: Seq) -> u64 {
        value.0
    }
}

impl From<&Seq> for u64 {
    #[inline]
    fn from(value: &Seq) -> u64 {
        value.0
    }
}

impl FromPyObject<'_> for Seq {
    fn extract_bound(ob: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        Ok(Self(ob.extract::<u64>()?))
    }
}

impl<'py> IntoPyObject<'py> for Seq {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        self.0.into_bound_py_any(py)
    }
}

/// Worker operation errors.
// TODO: Make other exceptions like CallFunctionError, etc. serializable and
// send them back to the client through WorkerError.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, thiserror::Error)]
#[error("worker {worker_actor_id} error: {backtrace}")]
pub struct WorkerError {
    /// The message and/or stack trace of the error.
    pub backtrace: String,

    /// Actor id of the worker that had the error.
    // TODO: arguably at this level we only care about the rank
    pub worker_actor_id: ActorId,
}

/// Device operation failures.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, thiserror::Error)]
#[error("device {actor_id} error: {backtrace}")]
pub struct DeviceFailure {
    /// The message and/or stack trace of the error.
    pub backtrace: String,

    /// Address of the device that had the error.
    pub address: String,

    /// Actor id of the worker that had the error.
    // TODO: arguably at this level we only care about the rank
    pub actor_id: ActorId,
}

/// Controller messages. These define the contract that the controller has with the client
/// and workers.
#[derive(Handler, HandleClient, RefClient, Serialize, Deserialize, Debug, Named)]
pub enum ControllerMessage {
    /// Attach a client to the controller. This is used to send messages to the controller
    /// and allow the controller to send messages back to the client.
    Attach {
        /// The client actor that is being attached.
        client_actor: ActorRef<ClientActor>,

        /// The response to indicate if the client was successfully attached.
        #[reply]
        response_port: hyperactor::OncePortRef<()>,
    },

    /// Notify the controller of the dependencies for a worker operation with the same seq.
    /// It is the responsibility of the caller to ensure the seq is unique and strictly
    /// increasing and matches the right message. This will be used by the controller for
    /// history / data dependency tracking.
    /// TODO: Support mutates here as well for proper dep management
    Node {
        seq: Seq,
        /// The set of references defined (or re-defined) by the operation.
        /// These are the operation's outputs.
        defs: Vec<Ref>,
        /// The set of references used by the operation. These are the operation's inputs.
        uses: Vec<Ref>,
    },

    // Mark references as being dropped by the client: the client will never
    // use these references again. Doing so results in undefined behavior.
    DropRefs {
        refs: Vec<Ref>,
    },

    /// Send a message to the workers mapping to the ranks provided in the
    /// given slice. The message is serialized bytes with the underlying datatype being
    /// [`crate::worker::WorkerMessage`] and serialization has been done in a hyperactor
    /// compatible way i.e. using [`bincode`]. These bytes will be forwarded to
    /// the workers as is. This helps provide isolation between the controller and the
    /// workers and avoids the need to pay the cost to deserialize pytrees in the controller.
    Send {
        ranks: Ranks,
        message: Serialized,
    },

    /// Response to a [`crate::worker::WorkerMessage::CallFunction`] message if
    /// the function errored.
    RemoteFunctionFailed {
        seq: Seq,
        error: WorkerError,
    },

    /// Response to a [`crate::worker::WorkerMessage::RequestStatus`] message. The payload will
    /// be set to the seq provided in the original message + 1.
    // TODO: T212094401 take a ActorRef
    Status {
        seq: Seq,
        worker_actor_id: ActorId,
        controller: bool,
    },

    /// Response to a [`crate::worker::WorkerMessage::SendValue`] message, containing the
    /// requested value. The value is serialized as a `Serialized` and deserialization
    /// is the responsibility of the caller. It should be deserialized as
    /// [`monarch_types::PyTree<RValue>`] using the [`Serialized::deserialized`] method.
    FetchResult {
        seq: Seq,
        value: Result<Serialized, WorkerError>,
    },

    /// This is used in unit tests to get the first incomplete seq for each rank as captured
    /// by the controller.
    GetFirstIncompleteSeqsUnitTestsOnly {
        #[reply]
        response_port: hyperactor::OncePortRef<Vec<Seq>>,
    },

    /// The message to schedule next supervision check task on the controller.
    CheckSupervision {},

    /// Debugger message sent from a debugger to be forwarded back to the client.
    DebuggerMessage {
        debugger_actor_id: ActorId,
        action: DebuggerAction,
    },
}

hyperactor::alias!(ControllerActor, ControllerMessage);
