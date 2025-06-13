/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Configuration for Hyperactor.
//!
//! This module provides a centralized way to manage configuration settings for Hyperactor.
//! It uses the attrs system for type-safe, flexible configuration management that supports
//! environment variables, YAML files, and temporary modifications for tests.

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::RwLock;
use std::time::Duration;

use crate::attrs::Attrs;
use crate::attrs::declare_attr_key;

// Declare configuration keys using the attrs system
declare_attr_key!(
    CODEC_MAX_FRAME_LENGTH,
    usize,
    "Maximum frame length for codec"
);

declare_attr_key!(
    MESSAGE_DELIVERY_TIMEOUT,
    Duration,
    "Message delivery timeout"
);

declare_attr_key!(
    MESSAGE_ACK_TIME_INTERVAL,
    Duration,
    "Message acknowledgment interval"
);

declare_attr_key!(
    MESSAGE_ACK_EVERY_N_MESSAGES,
    u64,
    "Number of messages after which to send an acknowledgment"
);

declare_attr_key!(
    SPLIT_MAX_BUFFER_SIZE,
    usize,
    "Maximum buffer size for split port messages"
);

declare_attr_key!(
    IS_MANAGED_SUBPROCESS,
    bool,
    "Flag indicating if this is a managed subprocess"
);

/// Configuration builder for Hyperactor.
#[derive(Debug, Clone)]
pub struct Config {
    attrs: Attrs,
}

impl Default for Config {
    fn default() -> Self {
        let mut attrs = Attrs::new();

        // Set default values
        attrs.set(CODEC_MAX_FRAME_LENGTH, 8 * 1024 * 1024); // 8 MB
        attrs.set(MESSAGE_DELIVERY_TIMEOUT, Duration::from_secs(30));
        attrs.set(MESSAGE_ACK_TIME_INTERVAL, Duration::from_millis(500));
        attrs.set(MESSAGE_ACK_EVERY_N_MESSAGES, 1000u64);
        attrs.set(SPLIT_MAX_BUFFER_SIZE, 5usize);
        attrs.set(IS_MANAGED_SUBPROCESS, false);

        Self { attrs }
    }
}

impl Config {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Load codec max frame length
        if let Ok(val) = env::var("HYPERACTOR_CODEC_MAX_FRAME_LENGTH") {
            if let Ok(parsed) = val.parse::<usize>() {
                config.attrs.set(CODEC_MAX_FRAME_LENGTH, parsed);
            }
        }

        // Load message delivery timeout
        if let Ok(val) = env::var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS") {
            if let Ok(parsed) = val.parse::<u64>() {
                config
                    .attrs
                    .set(MESSAGE_DELIVERY_TIMEOUT, Duration::from_secs(parsed));
            }
        }

        // Load message ack time interval
        if let Ok(val) = env::var("HYPERACTOR_MESSAGE_ACK_TIME_INTERVAL_MS") {
            if let Ok(parsed) = val.parse::<u64>() {
                config
                    .attrs
                    .set(MESSAGE_ACK_TIME_INTERVAL, Duration::from_millis(parsed));
            }
        }

        // Load message ack every n messages
        if let Ok(val) = env::var("HYPERACTOR_MESSAGE_ACK_EVERY_N_MESSAGES") {
            if let Ok(parsed) = val.parse::<u64>() {
                config.attrs.set(MESSAGE_ACK_EVERY_N_MESSAGES, parsed);
            }
        }

        // Load split max buffer size
        if let Ok(val) = env::var("HYPERACTOR_SPLIT_MAX_BUFFER_SIZE") {
            if let Ok(parsed) = val.parse::<usize>() {
                config.attrs.set(SPLIT_MAX_BUFFER_SIZE, parsed);
            }
        }

        // Check if this is a managed subprocess
        config.attrs.set(
            IS_MANAGED_SUBPROCESS,
            env::var("HYPERACTOR_MANAGED_SUBPROCESS").is_ok(),
        );

        config
    }

    /// Load configuration from a YAML file
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self, anyhow::Error> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let attrs: Attrs = serde_yaml::from_str(&contents)?;
        Ok(Self { attrs })
    }

    /// Merge with another configuration, with the other taking precedence
    pub fn merge(&mut self, other: &Self) {
        // For each key in the other config, copy it to this config
        if let Some(value) = other.attrs.get(CODEC_MAX_FRAME_LENGTH) {
            self.attrs.set(CODEC_MAX_FRAME_LENGTH, *value);
        }
        if let Some(value) = other.attrs.get(MESSAGE_DELIVERY_TIMEOUT) {
            self.attrs.set(MESSAGE_DELIVERY_TIMEOUT, *value);
        }
        if let Some(value) = other.attrs.get(MESSAGE_ACK_TIME_INTERVAL) {
            self.attrs.set(MESSAGE_ACK_TIME_INTERVAL, *value);
        }
        if let Some(value) = other.attrs.get(MESSAGE_ACK_EVERY_N_MESSAGES) {
            self.attrs.set(MESSAGE_ACK_EVERY_N_MESSAGES, *value);
        }
        if let Some(value) = other.attrs.get(SPLIT_MAX_BUFFER_SIZE) {
            self.attrs.set(SPLIT_MAX_BUFFER_SIZE, *value);
        }
        if let Some(value) = other.attrs.get(IS_MANAGED_SUBPROCESS) {
            self.attrs.set(IS_MANAGED_SUBPROCESS, *value);
        }
    }

    /// Save configuration to a YAML file
    pub fn to_yaml<P: AsRef<Path>>(&self, path: P) -> Result<(), anyhow::Error> {
        let yaml = serde_yaml::to_string(&self.attrs)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }

    /// Get the underlying attrs
    pub fn attrs(&self) -> &Attrs {
        &self.attrs
    }

    /// Get a mutable reference to the underlying attrs
    pub fn attrs_mut(&mut self) -> &mut Attrs {
        &mut self.attrs
    }

    // Convenience getter methods
    /// Get the codec max frame length
    pub fn codec_max_frame_length(&self) -> usize {
        *self
            .attrs
            .get(CODEC_MAX_FRAME_LENGTH)
            .unwrap_or(&(8 * 1024 * 1024))
    }

    /// Get the message delivery timeout
    pub fn message_delivery_timeout(&self) -> Duration {
        *self
            .attrs
            .get(MESSAGE_DELIVERY_TIMEOUT)
            .unwrap_or(&Duration::from_secs(30))
    }

    /// Get the message acknowledgment time interval
    pub fn message_ack_time_interval(&self) -> Duration {
        *self
            .attrs
            .get(MESSAGE_ACK_TIME_INTERVAL)
            .unwrap_or(&Duration::from_millis(500))
    }

    /// Get the number of messages after which to send an acknowledgment
    pub fn message_ack_every_n_messages(&self) -> u64 {
        *self
            .attrs
            .get(MESSAGE_ACK_EVERY_N_MESSAGES)
            .unwrap_or(&1000)
    }

    /// Get the maximum buffer size for split port messages
    pub fn split_max_buffer_size(&self) -> usize {
        *self.attrs.get(SPLIT_MAX_BUFFER_SIZE).unwrap_or(&5)
    }

    /// Check if this is a managed subprocess
    pub fn is_managed_subprocess(&self) -> bool {
        *self.attrs.get(IS_MANAGED_SUBPROCESS).unwrap_or(&false)
    }

    // Convenience setter methods
    /// Set the codec max frame length
    pub fn set_codec_max_frame_length(&mut self, value: usize) {
        self.attrs.set(CODEC_MAX_FRAME_LENGTH, value);
    }

    /// Set the message delivery timeout
    pub fn set_message_delivery_timeout(&mut self, value: Duration) {
        self.attrs.set(MESSAGE_DELIVERY_TIMEOUT, value);
    }

    /// Set the message acknowledgment time interval
    pub fn set_message_ack_time_interval(&mut self, value: Duration) {
        self.attrs.set(MESSAGE_ACK_TIME_INTERVAL, value);
    }

    /// Set the number of messages after which to send an acknowledgment
    pub fn set_message_ack_every_n_messages(&mut self, value: u64) {
        self.attrs.set(MESSAGE_ACK_EVERY_N_MESSAGES, value);
    }

    /// Set the maximum buffer size for split port messages
    pub fn set_split_max_buffer_size(&mut self, value: usize) {
        self.attrs.set(SPLIT_MAX_BUFFER_SIZE, value);
    }

    /// Set whether this is a managed subprocess
    pub fn set_is_managed_subprocess(&mut self, value: bool) {
        self.attrs.set(IS_MANAGED_SUBPROCESS, value);
    }
}

/// Global configuration functions
///
/// This module provides global configuration access and testing utilities.
///
/// # Testing with Global Configuration
///
/// Tests can override global configuration using [`global::lock`]. This ensures that
/// such tests are serialized (and cannot clobber each other's overrides).
///
/// ```ignore rust
/// #[test]
/// fn test_my_feature() {
///     let config = hyperactor::config::global::lock();
///     let _guard = config.override_key(SOME_CONFIG_KEY, test_value);
///     // ... test logic here ...
/// }
/// ```
pub mod global {
    use std::marker::PhantomData;

    use super::*;

    /// Global configuration instance, initialized from environment variables.
    static CONFIG: LazyLock<Arc<RwLock<Config>>> =
        LazyLock::new(|| Arc::new(RwLock::new(Config::from_env())));

    /// Acquire the global configuration lock for testing.
    ///
    /// This function returns a ConfigLock that acts as both a write lock guard (preventing
    /// other tests from modifying global config concurrently) and as the only way to
    /// create configuration overrides.
    ///
    /// Example usage:
    /// ```ignore rust
    /// let config = hyperactor::config::global::lock();
    /// let _guard = config.override_key(CONFIG_KEY, "value");
    /// // ... test code using the overridden config ...
    /// ```
    pub fn lock() -> ConfigLock {
        static MUTEX: LazyLock<std::sync::Mutex<()>> = LazyLock::new(|| std::sync::Mutex::new(()));
        ConfigLock {
            _guard: MUTEX.lock().unwrap(),
        }
    }

    /// Initialize the global configuration from environment variables
    pub fn init_from_env() {
        let config = Config::from_env();
        let mut global_config = CONFIG.write().unwrap();
        *global_config = config;
    }

    /// Initialize the global configuration from a YAML file
    pub fn init_from_yaml<P: AsRef<Path>>(path: P) -> Result<(), anyhow::Error> {
        let config = Config::from_yaml(path)?;
        let mut global_config = CONFIG.write().unwrap();
        *global_config = config;
        Ok(())
    }

    /// Get a reference to the global configuration
    pub fn get() -> Arc<RwLock<Config>> {
        CONFIG.clone()
    }

    /// Get the global attrs
    pub fn attrs() -> Attrs {
        CONFIG.read().unwrap().attrs.clone()
    }

    /// Get the codec max frame length
    pub fn codec_max_frame_length() -> usize {
        CONFIG.read().unwrap().codec_max_frame_length()
    }

    /// Get the message delivery timeout
    pub fn message_delivery_timeout() -> Duration {
        CONFIG.read().unwrap().message_delivery_timeout()
    }

    /// Get the message acknowledgment time interval
    pub fn message_ack_time_interval() -> Duration {
        CONFIG.read().unwrap().message_ack_time_interval()
    }

    /// Get the number of messages after which to send an acknowledgment
    pub fn message_ack_every_n_messages() -> u64 {
        CONFIG.read().unwrap().message_ack_every_n_messages()
    }

    /// Get the maximum buffer size for split port messages
    pub fn split_max_buffer_size() -> usize {
        CONFIG.read().unwrap().split_max_buffer_size()
    }

    /// Check if this is a managed subprocess
    pub fn is_managed_subprocess() -> bool {
        CONFIG.read().unwrap().is_managed_subprocess()
    }

    /// Reset the global configuration to defaults (for testing only)
    ///
    /// Note: This should be called from within with_test_lock() to ensure thread safety.
    /// Available in all builds to support tests in other crates.
    pub fn reset_to_defaults() {
        let mut config = CONFIG.write().unwrap();
        *config = Config::default();
    }

    /// A guard that holds the global configuration lock and provides override functionality.
    ///
    /// This struct acts as both a lock guard (preventing other tests from modifying global config)
    /// and as the only way to create configuration overrides. Override guards cannot outlive
    /// this ConfigLock, ensuring proper synchronization.
    pub struct ConfigLock {
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl ConfigLock {
        /// Create a configuration override that will be restored when the guard is dropped.
        ///
        /// The returned guard must not outlive this ConfigLock.
        pub fn override_key<
            'a,
            T: Send
                + Sync
                + serde::Serialize
                + serde::de::DeserializeOwned
                + crate::data::Named
                + Clone
                + 'static,
        >(
            &'a self,
            key: crate::attrs::Key<T>,
            value: T,
        ) -> ConfigValueGuard<'a, T> {
            let orig = {
                let mut config = CONFIG.write().unwrap();
                let orig = config.attrs.take_value(key);
                config.attrs.set(key, value);
                orig
            };

            ConfigValueGuard {
                key,
                orig,
                _phantom: PhantomData,
            }
        }
    }

    /// A guard that restores a single configuration value when dropped
    pub struct ConfigValueGuard<'a, T> {
        key: crate::attrs::Key<T>,
        orig: Option<Box<dyn crate::attrs::SerializableValue>>,
        // This is here so we can hold onto a 'a lifetime.
        _phantom: PhantomData<&'a ()>,
    }

    impl<T> Drop for ConfigValueGuard<'_, T> {
        fn drop(&mut self) {
            let mut config = CONFIG.write().unwrap();
            if let Some(orig) = self.orig.take() {
                config.attrs.restore_value(self.key, orig);
            } else {
                config.attrs.remove_value(&self.key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.codec_max_frame_length(), 8 * 1024 * 1024);
        assert_eq!(config.message_delivery_timeout(), Duration::from_secs(30));
        assert_eq!(
            config.message_ack_time_interval(),
            Duration::from_millis(500)
        );
        assert_eq!(config.message_ack_every_n_messages(), 1000);
        assert_eq!(config.split_max_buffer_size(), 5);
        assert!(!config.is_managed_subprocess());
    }

    #[test]
    fn test_from_env() {
        // Set environment variables
        // SAFETY: TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", "1024") };
        // SAFETY: TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS", "60") };

        let config = Config::from_env();

        assert_eq!(config.codec_max_frame_length(), 1024);
        assert_eq!(config.message_delivery_timeout(), Duration::from_secs(60));
        assert_eq!(
            config.message_ack_time_interval(),
            Duration::from_millis(500)
        ); // Default value

        // Clean up
        // SAFETY: TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("HYPERACTOR_CODEC_MAX_FRAME_LENGTH") };
        // SAFETY: TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS") };
    }

    #[test]
    fn test_merge() {
        let mut config1 = Config::default();
        let mut config2 = Config::default();
        config2.set_codec_max_frame_length(1024);
        config2.set_message_delivery_timeout(Duration::from_secs(60));

        config1.merge(&config2);

        assert_eq!(config1.codec_max_frame_length(), 1024);
        assert_eq!(config1.message_delivery_timeout(), Duration::from_secs(60));
    }

    #[test]
    fn test_global_config() {
        let config = global::lock();

        // Reset global config to defaults to avoid interference from other tests
        global::reset_to_defaults();

        assert_eq!(global::codec_max_frame_length(), 8 * 1024 * 1024);

        // Temporarily modify the configuration using the new lock/override API
        {
            let _guard = config.override_key(CODEC_MAX_FRAME_LENGTH, 1024);
            assert_eq!(global::codec_max_frame_length(), 1024);
        }

        // Check that the original configuration is restored
        assert_eq!(global::codec_max_frame_length(), 8 * 1024 * 1024);

        // Temporarily modify the configuration using the RAII pattern
        {
            let _guard = config.override_key(CODEC_MAX_FRAME_LENGTH, 1024);
            assert_eq!(global::codec_max_frame_length(), 1024);

            // The configuration will be automatically restored when _guard goes out of scope
        }

        // Check that the original configuration is restored
        assert_eq!(global::codec_max_frame_length(), 8 * 1024 * 1024);
    }

    #[test]
    fn test_overrides() {
        let config = global::lock();

        // Reset global config to defaults to avoid interference from other tests
        global::reset_to_defaults();

        // Test the new lock/override API for individual config values
        assert_eq!(global::codec_max_frame_length(), 8 * 1024 * 1024);
        assert_eq!(global::message_delivery_timeout(), Duration::from_secs(30));

        // Test single value override
        {
            let _guard = config.override_key(CODEC_MAX_FRAME_LENGTH, 2048);
            assert_eq!(global::codec_max_frame_length(), 2048);
            assert_eq!(global::message_delivery_timeout(), Duration::from_secs(30)); // Unchanged
        }

        // Values should be restored after guard is dropped
        assert_eq!(global::codec_max_frame_length(), 8 * 1024 * 1024);

        // Test multiple overrides
        {
            let _guard1 = config.override_key(CODEC_MAX_FRAME_LENGTH, 4096);
            let _guard2 = config.override_key(MESSAGE_DELIVERY_TIMEOUT, Duration::from_secs(60));

            assert_eq!(global::codec_max_frame_length(), 4096);
            assert_eq!(global::message_delivery_timeout(), Duration::from_secs(60));
        }

        // All values should be restored
        assert_eq!(global::codec_max_frame_length(), 8 * 1024 * 1024);
        assert_eq!(global::message_delivery_timeout(), Duration::from_secs(30));
    }
}
