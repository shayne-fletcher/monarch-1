# Actors

Hyperactor programs are structured around actors: isolated state machines that process messages asynchronously.

Each actor runs in isolation, and maintains private internal state. Actors interact with the outside world through typed message ports and follow strict lifecycle semantics managed by the runtime.

This chapter introduces the actor system in hyperactor. We'll cover:

- The [`Actor`](./actor.md) trait and its lifecycle hooks
- The [`Handler`](./handler.md) trait for defining message-handling behavior
- The [`RemotableActor`](./remotable_actor.md) trait for enabling remote spawning
- The [`Checkpointable`](./checkpointable.md) trait for supporting actor persistence and recovery
- The [`RemoteActor`](./remote_actor.md) marker trait for remotely referencable types
- The [`Binds`](./binds.md) trait for wiring exported ports to reference types
- The [`RemoteHandles`](./remote_handles.md) trait for associating message types with a reference

Actors are always instantiated with parameters and bound to a mailbox, enabling them to participate in reliable message-passing systems. Supervision, checkpointing, and references all build upon this core abstraction.
