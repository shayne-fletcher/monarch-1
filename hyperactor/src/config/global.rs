/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Global layered configuration for Hyperactor.
//!
//! This module provides the process-wide configuration store and APIs
//! to access it. Configuration values are resolved via a **layered
//! model**: `TestOverride → Runtime → Env → File → Default`.
//!
//! - Reads (`get`, `get_cloned`) consult layers in that order, falling
//!   back to defaults if no explicit value is set.
//! - `attrs()` returns a **complete snapshot** of the effective
//!   configuration at call time: it materializes defaults for keys not
//!   set in any layer, and omits meta-only keys (like `CONFIG_ENV_VAR`)
//!   unless explicitly set.
//! - In tests, `lock()` and `override_key` allow temporary overrides
//!   that are removed automatically when the guard drops.
//! - In normal operation, a parent process can capture its effective
//!   config via `attrs()` and pass that snapshot to a child during
//!   bootstrap. The child installs it as a `Runtime` layer so the
//!   parent’s values take precedence over Env/File/Defaults.
//!
//! This design provides flexibility (easy test overrides, runtime
//! updates, YAML/Env baselines) while ensuring type safety and
//! predictable resolution order.
//!
//!
//! # Testing
//!
//! Tests can override global configuration using [`lock`]. This
//! ensures such tests are serialized (and cannot clobber each other's
//! overrides).
//!
//! ```ignore
//! #[test]
//! fn test_my_feature() {
//!     let config = hyperactor::config::global::lock();
//!     let _guard = config.override_key(SOME_CONFIG_KEY, test_value);
//!     // ... test logic here ...
//! }
//! ```
use std::marker::PhantomData;

use super::*;
use crate::attrs::AttrValue;
use crate::attrs::Key;

/// Configuration source layers in priority order.
///
/// Resolution order is always: **TestOverride -> Runtime -> Env
/// -> File -> Default**.
///
/// Smaller `priority()` number = higher precedence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// Values loaded from configuration files (e.g., YAML). This
    /// is the lowest-priority explicit source.
    File,
    /// Values read from environment variables at process startup.
    /// Higher priority than File, but lower than
    /// Runtime/TestOverride.
    Env,
    /// Values set programmatically at runtime. Highest stable
    /// priority layer; only overridden by TestOverride.
    Runtime,
    /// Ephemeral values inserted by tests via
    /// `ConfigLock::override_key`. Always wins over all other
    /// sources; removed when the guard drops.
    TestOverride,
}

/// Return the numeric priority for a source.
///
/// Smaller number = higher precedence. Matches the documented
/// order: TestOverride (0) -> Runtime (1) -> Env (2) -> File (3).
fn priority(s: Source) -> u8 {
    match s {
        Source::TestOverride => 0,
        Source::Runtime => 1,
        Source::Env => 2,
        Source::File => 3,
    }
}

/// A single configuration layer in the global store.
///
/// Each `Layer` wraps a [`Source`] and its associated [`Attrs`]
/// values. Layers are kept in priority order and consulted during
/// resolution.
#[derive(Clone)]
struct Layer {
    /// The origin of this layer (File, Env, Runtime, or
    /// TestOverride).
    source: Source,
    /// The set of attributes explicitly provided by this source.
    attrs: Attrs,
}

/// The full set of configuration layers in priority order.
///
/// `Layers` wraps a vector of [`Layer`]s, always kept sorted by
/// [`priority`] (lowest number = highest precedence).
///
/// Resolution (`get`, `get_cloned`, `attrs`) consults `ordered`
/// from front to back, returning the first value found for each
/// key and falling back to defaults if none are set in any layer.
struct Layers {
    /// Kept sorted by `priority` (lowest number first = highest
    /// priority).
    ordered: Vec<Layer>,
}

/// Global layered configuration store.
///
/// This is the single authoritative store for configuration in
/// the process. It is always present, protected by an `RwLock`,
/// and holds a [`Layers`] struct containing all active sources.
///
/// On startup it is seeded with a single [`Source::Env`] layer
/// (values loaded from process environment variables). Additional
/// layers can be installed later via [`set`] or cleared with
/// [`clear`]. Reads (`get`, `get_cloned`, `attrs`) consult the
/// layers in priority order.
///
/// In tests, a [`Source::TestOverride`] layer is pushed on demand
/// by [`ConfigLock::override_key`]. This layer always takes
/// precedence and is automatically removed when the guard drops.
///
/// In normal operation, a parent process may capture its config
/// with [`attrs`] and pass it to a child during bootstrap. The
/// child installs this snapshot as its [`Source::Runtime`] layer,
/// ensuring the parent's values override Env/File/Defaults.
static LAYERS: LazyLock<Arc<RwLock<Layers>>> = LazyLock::new(|| {
    let env = super::from_env();
    let layers = Layers {
        ordered: vec![Layer {
            source: Source::Env,
            attrs: env,
        }],
    };
    Arc::new(RwLock::new(layers))
});

/// Acquire the global configuration lock.
///
/// This lock serializes all mutations of the global
/// configuration, ensuring they cannot clobber each other. It
/// returns a [`ConfigLock`] guard, which must be held for the
/// duration of any mutation (e.g. inserting or overriding
/// values).
///
/// Most commonly used in tests, where it provides exclusive
/// access to push a [`Source::TestOverride`] layer via
/// [`ConfigLock::override_key`]. The override layer is
/// automatically removed when the guard drops, restoring the
/// original state.
///
/// # Example
/// ```rust,ignore
/// let lock = hyperactor::config::global::lock();
/// let _guard = lock.override_key(CONFIG_KEY, "test_value");
/// // Code under test sees the overridden config.
/// // On drop, the key is restored.
/// ```
pub fn lock() -> ConfigLock {
    static MUTEX: LazyLock<std::sync::Mutex<()>> = LazyLock::new(|| std::sync::Mutex::new(()));
    ConfigLock {
        _guard: MUTEX.lock().unwrap(),
    }
}

/// Initialize the global configuration from environment
/// variables.
///
/// Reads values from process environment variables, using the
/// `CONFIG_ENV_VAR` meta-attribute declared on each key to find
/// its mapping. The resulting values are installed as the
/// [`Source::Env`] layer. Keys without a corresponding
/// environment variable fall back to defaults or higher-priority
/// sources.
///
/// Typically invoked once at process startup to overlay config
/// values from the environment. Repeated calls replace the
/// existing Env layer.
pub fn init_from_env() {
    set(Source::Env, super::from_env());
}

/// Initialize the global configuration from a YAML file.
///
/// Loads values from the specified YAML file and installs them as
/// the [`Source::File`] layer. This is the lowest-priority
/// explicit source: values from Env, Runtime, or TestOverride
/// layers always take precedence. Keys not present in the file
/// fall back to their defaults or higher-priority sources.
///
/// Typically invoked once at process startup to provide a
/// baseline configuration. Repeated calls replace the existing
/// File layer.
pub fn init_from_yaml<P: AsRef<Path>>(path: P) -> Result<(), anyhow::Error> {
    let file = super::from_yaml(path)?;
    set(Source::File, file);
    Ok(())
}

/// Get a key from the global configuration (Copy types).
///
/// Resolution order: TestOverride -> Runtime -> Env -> File ->
/// Default. Panics if the key has no default and is not set in
/// any layer.
pub fn get<T: AttrValue + Copy>(key: Key<T>) -> T {
    let layers = LAYERS.read().unwrap();
    for layer in &layers.ordered {
        if layer.attrs.contains_key(key) {
            return *layer.attrs.get(key).unwrap();
        }
    }
    *key.default().expect("key must have a default")
}

/// Get a key by cloning the value.
///
/// Resolution order: TestOverride -> Runtime -> Env -> File ->
/// Default. Panics if the key has no default and is not set in
/// any layer.
pub fn get_cloned<T: AttrValue>(key: Key<T>) -> T {
    try_get_cloned(key)
        .expect("key must have a default")
        .clone()
}

/// Try to get a key by cloning the value.
///
/// Resolution order: TestOverride -> Runtime -> Env -> File ->
/// Default. Returns None if the key has no default and is not set in
/// any layer.
pub fn try_get_cloned<T: AttrValue>(key: Key<T>) -> Option<T> {
    let layers = LAYERS.read().unwrap();
    for layer in &layers.ordered {
        if layer.attrs.contains_key(key) {
            return layer.attrs.get(key).cloned();
        }
    }
    key.default().cloned()
}

/// Insert or replace a configuration layer for the given source.
///
/// If a layer with the same [`Source`] already exists, its
/// contents are replaced with the provided `attrs`. Otherwise a
/// new layer is added. After insertion, layers are re-sorted so
/// that higher-priority sources (e.g. [`Source::TestOverride`],
/// [`Source::Runtime`]) appear before lower-priority ones
/// ([`Source::Env`], [`Source::File`]).
///
/// This function is used by initialization routines (e.g.
/// `init_from_env`, `init_from_yaml`) and by tests when
/// overriding configuration values.
pub fn set(source: Source, attrs: Attrs) {
    let mut g = LAYERS.write().unwrap();
    if let Some(l) = g.ordered.iter_mut().find(|l| l.source == source) {
        l.attrs = attrs;
    } else {
        g.ordered.push(Layer { source, attrs });
    }
    g.ordered.sort_by_key(|l| priority(l.source)); // TestOverride < Runtime < Env < File
}

/// Remove the configuration layer for the given [`Source`], if
/// present.
///
/// After this call, values from that source will no longer
/// contribute to resolution in [`get`], [`get_cloned`], or
/// [`attrs`]. Defaults and any remaining layers continue to apply
/// in their normal priority order.
pub(crate) fn clear(source: Source) {
    let mut g = LAYERS.write().unwrap();
    g.ordered.retain(|l| l.source != source);
}

/// Return a complete, merged snapshot of the effective
/// configuration.
///
/// Resolution per key:
/// 1) First explicit value found in layers (TestOverride →
///    Runtime → Env → File).
/// 2) Otherwise, the key's default (if any).
///
/// Notes:
/// - This materializes defaults into the returned Attrs so it's
///   self-contained.
pub fn attrs() -> Attrs {
    let layers = LAYERS.read().unwrap();
    let mut merged = Attrs::new();

    // Iterate all declared keys (registered via `declare_attrs!`
    // + inventory).
    for info in inventory::iter::<AttrKeyInfo>() {
        let name = info.name;

        // Try to resolve from highest -> lowest priority layer.
        let mut chosen: Option<Box<dyn crate::attrs::SerializableValue>> = None;
        for layer in &layers.ordered {
            if let Some(v) = layer.attrs.get_value_by_name(name) {
                chosen = Some(v.cloned());
                break;
            }
        }

        // If no explicit value, materialize the default if there
        // is one.
        let boxed = match chosen {
            Some(b) => b,
            None => {
                if let Some(default) = info.default {
                    default.cloned()
                } else {
                    // No explicit value and no default — skip
                    // this key.
                    continue;
                }
            }
        };

        merged.insert_value_by_name_unchecked(name, boxed);
    }

    merged
}

/// Reset the global configuration to only Defaults (for testing).
///
/// This clears all explicit layers (`File`, `Env`, `Runtime`, and
/// `TestOverride`). Subsequent lookups will resolve keys entirely
/// from their declared defaults.
///
/// Note: Should be called while holding [`global::lock`] in
/// tests, to ensure no concurrent modifications happen.
pub fn reset_to_defaults() {
    let mut g = LAYERS.write().unwrap();
    g.ordered.clear();
}

fn test_override_index(layers: &Layers) -> Option<usize> {
    layers
        .ordered
        .iter()
        .position(|l| matches!(l.source, Source::TestOverride))
}

fn ensure_test_override_layer_mut(layers: &mut Layers) -> &mut Attrs {
    if let Some(i) = test_override_index(layers) {
        return &mut layers.ordered[i].attrs;
    }
    layers.ordered.push(Layer {
        source: Source::TestOverride,
        attrs: Attrs::new(),
    });
    layers.ordered.sort_by_key(|l| priority(l.source));
    let i = test_override_index(layers).expect("just inserted TestOverride layer");
    &mut layers.ordered[i].attrs
}

/// A guard that holds the global configuration lock and provides
/// override functionality.
///
/// This struct acts as both a lock guard (preventing other tests
/// from modifying global config) and as the only way to create
/// configuration overrides. Override guards cannot outlive this
/// ConfigLock, ensuring proper synchronization.
pub struct ConfigLock {
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl ConfigLock {
    /// Create a configuration override that will be restored when
    /// the guard is dropped.
    ///
    /// The returned guard must not outlive this ConfigLock.
    pub fn override_key<'a, T: AttrValue>(
        &'a self,
        key: crate::attrs::Key<T>,
        value: T,
    ) -> ConfigValueGuard<'a, T> {
        // Write into the single TestOverride layer (create if
        // needed).
        let (prev_in_layer, orig_env) = {
            let mut guard = LAYERS.write().unwrap();
            let layer_attrs = ensure_test_override_layer_mut(&mut guard);
            // Save any previous override for this key in the the
            // TestOverride layer.
            let prev = layer_attrs.remove_value(key);
            // Set new override value.
            layer_attrs.set(key, value.clone());
            // Mirror env var.
            let orig_env = if let Some(env_var) = key.attrs().get(CONFIG_ENV_VAR) {
                let orig = std::env::var(env_var).ok();
                // SAFETY: Mutating process-global environment
                // variables is not thread-safe. This path is used
                // only in tests while holding the global
                // ConfigLock, which serializes config mutations
                // across the process. Tests are single-threaded
                // with respect to env changes, so there are no
                // concurrent readers/writers. We also record the
                // original value and restore it in
                // ConfigValueGuard::drop.
                unsafe {
                    std::env::set_var(env_var, value.display());
                }
                Some((env_var.clone(), orig))
            } else {
                None
            };
            (prev, orig_env)
        };

        ConfigValueGuard {
            key,
            orig: prev_in_layer, // previous value for this key *inside* TestOverride layer
            orig_env,
            _phantom: PhantomData,
        }
    }
}

/// When a [`ConfigLock`] is dropped, the special
/// [`Source::TestOverride`] layer (if present) is removed
/// entirely. This discards all temporary overrides created under
/// the lock, ensuring they cannot leak into subsequent tests or
/// callers. Other layers (`Runtime`, `Env`, `File`, and defaults)
/// are left untouched.
///
/// Note: individual values within the TestOverride layer may
/// already have been restored by [`ConfigValueGuard`]s as they
/// drop. This final removal guarantees no residual layer remains
/// once the lock itself is released.
impl Drop for ConfigLock {
    fn drop(&mut self) {
        let mut guard = LAYERS.write().unwrap();
        if let Some(pos) = guard
            .ordered
            .iter()
            .position(|l| matches!(l.source, Source::TestOverride))
        {
            guard.ordered.remove(pos);
        }
        // No need to restore anything else; underlying layers
        // remain intact.
    }
}

/// A guard that restores a single configuration value when dropped
pub struct ConfigValueGuard<'a, T: 'static> {
    key: crate::attrs::Key<T>,
    orig: Option<Box<dyn crate::attrs::SerializableValue>>,
    orig_env: Option<(String, Option<String>)>,
    // This is here so we can hold onto a 'a lifetime.
    _phantom: PhantomData<&'a ()>,
}

/// When a [`ConfigValueGuard`] is dropped, it restores the
/// configuration state for the key it was guarding:
///
/// - If there was a previous override for this key in the
///   [`Source::TestOverride`] layer, that value is reinserted.
/// - If this guard was the only override for the key, the entry
///   is removed from the layer entirely (leaving underlying layers
///   or defaults to apply).
/// - If the key declared a `CONFIG_ENV_VAR`, the corresponding
///   process environment variable is restored to its original value
///   (or removed if it didn't exist).
///
/// This ensures that overrides applied via
/// [`ConfigLock::override_key`] are always reverted cleanly when
/// the guard is dropped, without leaking state into subsequent
/// tests or callers.
impl<T: 'static> Drop for ConfigValueGuard<'_, T> {
    fn drop(&mut self) {
        let mut guard = LAYERS.write().unwrap();

        if let Some(i) = test_override_index(&guard) {
            let layer_attrs = &mut guard.ordered[i].attrs;

            if let Some(prev) = self.orig.take() {
                layer_attrs.insert_value(self.key, prev);
            } else {
                // remove without needing T: AttrValue
                let _ = layer_attrs.remove_value(self.key);
            }
        }

        if let Some((k, v)) = self.orig_env.take() {
            // SAFETY: process-global environment variables are
            // not thread-safe to mutate. This override/restore
            // path is only ever used in single-threaded test
            // code, and is serialized by the global ConfigLock to
            // avoid races between tests.
            unsafe {
                if let Some(v) = v {
                    std::env::set_var(k, v);
                } else {
                    std::env::remove_var(&k);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_global_config() {
        let config = lock();

        // Reset global config to defaults to avoid interference from other tests
        reset_to_defaults();

        assert_eq!(get(CODEC_MAX_FRAME_LENGTH), CODEC_MAX_FRAME_LENGTH_DEFAULT);
        {
            let _guard = config.override_key(CODEC_MAX_FRAME_LENGTH, 1024);
            assert_eq!(get(CODEC_MAX_FRAME_LENGTH), 1024);
            // The configuration will be automatically restored when _guard goes out of scope
        }

        assert_eq!(get(CODEC_MAX_FRAME_LENGTH), CODEC_MAX_FRAME_LENGTH_DEFAULT);
    }

    #[test]
    fn test_overrides() {
        let config = lock();

        // Reset global config to defaults to avoid interference from other tests
        reset_to_defaults();

        // Test the new lock/override API for individual config values
        assert_eq!(get(CODEC_MAX_FRAME_LENGTH), CODEC_MAX_FRAME_LENGTH_DEFAULT);
        assert_eq!(get(MESSAGE_DELIVERY_TIMEOUT), Duration::from_secs(30));

        // Test single value override
        {
            let _guard = config.override_key(CODEC_MAX_FRAME_LENGTH, 2048);
            assert_eq!(get(CODEC_MAX_FRAME_LENGTH), 2048);
            assert_eq!(get(MESSAGE_DELIVERY_TIMEOUT), Duration::from_secs(30)); // Unchanged
        }

        // Values should be restored after guard is dropped
        assert_eq!(get(CODEC_MAX_FRAME_LENGTH), CODEC_MAX_FRAME_LENGTH_DEFAULT);

        // Test multiple overrides
        let orig_value = std::env::var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT").ok();
        {
            let _guard1 = config.override_key(CODEC_MAX_FRAME_LENGTH, 4096);
            let _guard2 = config.override_key(MESSAGE_DELIVERY_TIMEOUT, Duration::from_secs(60));

            assert_eq!(get(CODEC_MAX_FRAME_LENGTH), 4096);
            assert_eq!(get(MESSAGE_DELIVERY_TIMEOUT), Duration::from_secs(60));
            // This was overridden:
            assert_eq!(
                std::env::var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT").unwrap(),
                "1m"
            );
        }
        assert_eq!(
            std::env::var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT").ok(),
            orig_value
        );

        // All values should be restored
        assert_eq!(get(CODEC_MAX_FRAME_LENGTH), CODEC_MAX_FRAME_LENGTH_DEFAULT);
        assert_eq!(get(MESSAGE_DELIVERY_TIMEOUT), Duration::from_secs(30));
    }

    #[test]
    fn test_layer_precedence_env_over_file_and_replacement() {
        let _lock = lock();
        reset_to_defaults();

        // File sets a value.
        let mut file = Attrs::new();
        file[CODEC_MAX_FRAME_LENGTH] = 1111;
        set(Source::File, file);

        // Env sets a different value.
        let mut env = Attrs::new();
        env[CODEC_MAX_FRAME_LENGTH] = 2222;
        set(Source::Env, env);

        // Env should win over File.
        assert_eq!(get(CODEC_MAX_FRAME_LENGTH), 2222);

        // Replace Env layer with a new value.
        let mut env2 = Attrs::new();
        env2[CODEC_MAX_FRAME_LENGTH] = 3333;
        set(Source::Env, env2);

        assert_eq!(get(CODEC_MAX_FRAME_LENGTH), 3333);
    }

    #[test]
    fn test_runtime_overrides_and_clear_restores_lower_layers() {
        let _lock = lock();
        reset_to_defaults();

        // File baseline.
        let mut file = Attrs::new();
        file[MESSAGE_DELIVERY_TIMEOUT] = Duration::from_secs(30);
        set(Source::File, file);

        // Env override.
        let mut env = Attrs::new();
        env[MESSAGE_DELIVERY_TIMEOUT] = Duration::from_secs(40);
        set(Source::Env, env);

        // Runtime beats both.
        let mut rt = Attrs::new();
        rt[MESSAGE_DELIVERY_TIMEOUT] = Duration::from_secs(50);
        set(Source::Runtime, rt);

        assert_eq!(get(MESSAGE_DELIVERY_TIMEOUT), Duration::from_secs(50));

        // Clearing Runtime should reveal Env again.
        clear(Source::Runtime);

        // With the Runtime layer gone, Env still wins over File.
        assert_eq!(get(MESSAGE_DELIVERY_TIMEOUT), Duration::from_secs(40));
    }

    #[test]
    fn test_attrs_snapshot_materializes_defaults_and_omits_meta() {
        let _lock = lock();
        reset_to_defaults();

        // No explicit layers: values should come from Defaults.
        let snap = attrs();

        // A few representative defaults are materialized:
        assert_eq!(snap[CODEC_MAX_FRAME_LENGTH], 10 * 1024 * 1024 * 1024);
        assert_eq!(snap[MESSAGE_DELIVERY_TIMEOUT], Duration::from_secs(30));

        // CONFIG_ENV_VAR has no default and wasn't explicitly set:
        // should be omitted.
        let json = serde_json::to_string(&snap).unwrap();
        assert!(
            !json.contains("config_env_var"),
            "CONFIG_ENV_VAR must not appear in snapshot unless explicitly set"
        );
    }

    #[test]
    fn test_parent_child_snapshot_as_runtime_layer() {
        let _lock = lock();
        reset_to_defaults();

        // Parent effective config (pretend it's a parent process).
        let mut parent_env = Attrs::new();
        parent_env[MESSAGE_ACK_EVERY_N_MESSAGES] = 12345;
        set(Source::Env, parent_env);

        let parent_snap = attrs();

        // "Child" process: start clean, install parent snapshot as
        // Runtime.
        reset_to_defaults();
        set(Source::Runtime, parent_snap);

        // Child should observe parent's effective value (as highest
        // stable layer).
        assert_eq!(get(MESSAGE_ACK_EVERY_N_MESSAGES), 12345);
    }

    #[test]
    fn test_testoverride_layer_override_and_env_restore() {
        let lock = lock();
        reset_to_defaults();

        assert_eq!(get(MESSAGE_DELIVERY_TIMEOUT), Duration::from_secs(30));

        // SAFETY: single-threaded test.
        unsafe {
            std::env::remove_var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT");
        }

        {
            let _guard = lock.override_key(MESSAGE_DELIVERY_TIMEOUT, Duration::from_secs(99));
            // Override wins:
            assert_eq!(get(MESSAGE_DELIVERY_TIMEOUT), Duration::from_secs(99));

            // Env should be mirrored to the same duration (string may
            // be "1m 39s")
            let s = std::env::var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT").unwrap();
            let parsed = humantime::parse_duration(&s).unwrap();
            assert_eq!(parsed, Duration::from_secs(99));
        }

        // After drop, value and env restored:
        assert_eq!(get(MESSAGE_DELIVERY_TIMEOUT), Duration::from_secs(30));
        assert!(std::env::var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT").is_err());
    }

    #[test]
    fn test_reset_to_defaults_clears_all_layers() {
        let _lock = lock();
        reset_to_defaults();

        // Seed multiple layers.
        let mut file = Attrs::new();
        file[SPLIT_MAX_BUFFER_SIZE] = 7;
        set(Source::File, file);

        let mut env = Attrs::new();
        env[SPLIT_MAX_BUFFER_SIZE] = 8;
        set(Source::Env, env);

        let mut rt = Attrs::new();
        rt[SPLIT_MAX_BUFFER_SIZE] = 9;
        set(Source::Runtime, rt);

        // Sanity: highest wins.
        assert_eq!(get(SPLIT_MAX_BUFFER_SIZE), 9);

        // Reset clears all explicit layers; defaults apply.
        reset_to_defaults();
        assert_eq!(get(SPLIT_MAX_BUFFER_SIZE), 5); // default
    }

    #[test]
    fn test_get_cloned_resolution_matches_get() {
        let _lock = lock();
        reset_to_defaults();

        let mut env = Attrs::new();
        env[CHANNEL_MULTIPART] = false;
        set(Source::Env, env);

        assert!(!get(CHANNEL_MULTIPART));
        let v = get_cloned(CHANNEL_MULTIPART);
        assert!(!v);
    }

    #[test]
    fn test_attrs_snapshot_respects_layer_precedence_per_key() {
        let _lock = lock();
        reset_to_defaults();

        let mut file = Attrs::new();
        file[MESSAGE_TTL_DEFAULT] = 10;
        set(Source::File, file);

        let mut env = Attrs::new();
        env[MESSAGE_TTL_DEFAULT] = 20;
        set(Source::Env, env);

        let snap = attrs();
        assert_eq!(snap[MESSAGE_TTL_DEFAULT], 20); // Env beats File
    }
}
