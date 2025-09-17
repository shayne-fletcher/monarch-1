/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::marker::PhantomData;

use hyperactor::Actor;
use hyperactor::ActorRef;
use hyperactor::RemoteHandles;
use hyperactor::RemoteMessage;
use hyperactor::actor::RemoteActor;
use hyperactor::context;
use hyperactor::message::Castable;
use ndslice::view;
use ndslice::view::Region;
use ndslice::view::View;
use ndslice::view::ViewExt;
use serde::Deserialize;
use serde::Serialize;

use crate::v1;
use crate::v1::Error;
use crate::v1::Name;
use crate::v1::ProcMeshRef;

/// An ActorMesh is a collection of ranked A-typed actors.
#[derive(Debug)]
pub struct ActorMesh<A> {
    proc_mesh: ProcMeshRef,
    name: Name,
    _phantom: PhantomData<A>,
}

impl<A> ActorMesh<A> {
    pub(crate) fn new(proc_mesh: ProcMeshRef, name: Name) -> Self {
        Self {
            proc_mesh,
            name,
            _phantom: PhantomData,
        }
    }
}

impl<A: RemoteActor> ActorMesh<A> {
    /// Freeze this actor mesh in its current state, returning a stable
    /// reference that may be serialized.
    pub fn freeze(&self) -> ActorMeshRef<A> {
        let actor_refs = self
            .proc_mesh
            .values()
            .map(|p| p.attest(&self.name))
            .collect();
        ActorMeshRef::new(self.name.clone(), self.proc_mesh.clone(), actor_refs)
    }
}

/// A reference to a stable snapshot of an [`ActorMesh`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActorMeshRef<A: RemoteActor> {
    proc_mesh: ProcMeshRef,
    name: Name,
    actor_refs: Vec<ActorRef<A>>, // Ranked::get() -> &ActorRef<A>
    _phantom: PhantomData<A>,
}

impl<A: RemoteActor> ActorMeshRef<A> {
    pub(crate) fn new(name: Name, proc_mesh: ProcMeshRef, actor_refs: Vec<ActorRef<A>>) -> Self {
        Self {
            proc_mesh,
            name,
            actor_refs,
            _phantom: PhantomData,
        }
    }
}

impl<A: Actor + RemoteActor> ActorMeshRef<A> {
    /// Cast a message to all actors in this mesh.
    pub fn cast<M>(&self, cx: &impl context::Actor, message: M) -> v1::Result<()>
    where
        M: Castable + RemoteMessage + Clone,
        A: RemoteHandles<M>,
    {
        // todo: headers, binding/unbinding/accumulation
        for actor_ref in self.values() {
            actor_ref
                .send(cx, message.clone())
                .map_err(|e| Error::SendingError(actor_ref.actor_id().clone(), Box::new(e)))?;
        }
        Ok(())
    }
}

impl<A: RemoteActor> view::Ranked for ActorMeshRef<A> {
    type Item = ActorRef<A>;

    fn region(&self) -> &Region {
        view::Ranked::region(&self.proc_mesh)
    }

    fn get(&self, rank: usize) -> Option<&Self::Item> {
        self.actor_refs.get(rank)
    }
}

impl<A: RemoteActor> view::RankedSliceable for ActorMeshRef<A> {
    fn sliced(&self, region: Region) -> Self {
        debug_assert!(region.is_subset(view::Ranked::region(self)));
        let proc_mesh = self.proc_mesh.subset(region.clone()).unwrap();
        let actor_refs = self
            .region()
            .remap(&region)
            .unwrap()
            .map(|index| self.actor_refs[index].clone())
            .collect();
        ActorMeshRef::new(self.name.clone(), proc_mesh, actor_refs)
    }
}
