/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TLS-aware `reqwest` client construction for the admin TUI.
//!
//! This module builds `(base_url, reqwest::Client)` from a
//! [`TuiConfig`], choosing HTTP vs HTTPS and configuring certificate
//! verification when TLS material is available.
//!
//! Admin handle resolution (`mast_conda:///` and bare `host:port`) is
//! performed in the binary via `AdminHandle::parse` before calling
//! this library. By the time [`build_client`] is called, `addr` is a
//! concrete URL: `https://host:port`, or `http://host:port` for an
//! explicit `http` scheme.
//!
//! # Address handling
//!
//! - `--addr` may be `host:port` (no scheme) or an explicit
//!   `http://...` / `https://...`.
//! - An explicit `https://` uses TLS; an explicit `http://` is a hint
//!   that auto-detected TLS material can still upgrade. `--plaintext`
//!   forces plain HTTP unconditionally.
//!
//! # TLS configuration (highest priority first)
//!
//! 0. `--plaintext`: disable TLS entirely and use plain HTTP.
//! 1. Explicit CLI paths: `--tls-ca` (required to enable TLS), with
//!    optional `--tls-cert` + `--tls-key` for mutual TLS.
//! 2. Auto-detection via [`hyperactor::channel::try_tls_pem_bundle`],
//!    which probes configured paths (OSS) and Meta well-known
//!    locations.
//! 3. Fallback to plain HTTP when no usable CA is found.
//!
//! **Note:** At Meta (`fbcode_build`), the mesh admin server requires
//! mutual TLS. If the client cannot load a CA certificate or fails to
//! parse a client identity, the connection will be rejected at the TLS
//! handshake. In OSS, the server falls back to plain HTTP when no
//! certs are available, so the client's HTTP fallback still works.

use crate::TuiConfig;

/// Read all bytes from a [`Pem`](hyperactor::config::Pem), returning
/// `None` if it can't be opened/read or if the result is empty.
fn read_pem(pem: &hyperactor::config::Pem) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut buf = Vec::new();
    pem.reader().ok()?.read_to_end(&mut buf).ok()?;
    if buf.is_empty() { None } else { Some(buf) }
}

/// Split an address into an optional `http`/`https` scheme and the
/// remaining host part.
///
/// If `addr` starts with `https://` or `http://`, returns
/// `(Some(scheme), rest)`. Otherwise returns `(None, addr)`. The
/// returned `&str` values are slices of `addr`.
fn parse_addr(addr: &str) -> (Option<&str>, &str) {
    if let Some(host) = addr.strip_prefix("https://") {
        (Some("https"), host)
    } else if let Some(host) = addr.strip_prefix("http://") {
        (Some("http"), host)
    } else {
        (None, addr)
    }
}

/// Configure TLS on a `reqwest::ClientBuilder` by adding a root CA,
/// and optionally a client identity (cert + key) for mutual TLS.
///
/// Delegates to [`hyperactor_mesh::mesh_admin_client::add_tls`] which
/// handles the fbcode/OSS native-tls vs rustls identity split.
fn add_tls(
    builder: reqwest::ClientBuilder,
    ca_bytes: &[u8],
    cert_bytes: Option<Vec<u8>>,
    key_bytes: Option<Vec<u8>>,
) -> (reqwest::ClientBuilder, bool) {
    hyperactor_mesh::mesh_admin_client::add_tls(builder, ca_bytes, cert_bytes, key_bytes)
}

/// Configure TLS on a `reqwest::ClientBuilder` using PEM files
/// supplied via CLI (`--tls-ca`, and optionally `--tls-cert` +
/// `--tls-key`).
///
/// Reads `ca_path` and installs it as the root trust anchor. If the
/// CA file cannot be read, returns `(builder, false)` with no
/// changes.
///
/// If `cert_path`/`key_path` are provided, attempts to read them and
/// pass the bytes through to [`add_tls`] to configure an mTLS
/// identity; failures to read these optional files simply omit the
/// identity (the CA may still be applied).
///
/// Returns `(updated_builder, ca_installed)`.
fn add_tls_from_paths(
    builder: reqwest::ClientBuilder,
    ca_path: &str,
    cert_path: Option<&str>,
    key_path: Option<&str>,
) -> (reqwest::ClientBuilder, bool) {
    let ca_bytes = match std::fs::read(ca_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("TLS: cannot read CA file {}: {}", ca_path, e);
            return (builder, false);
        }
    };
    let cert_bytes = cert_path.and_then(|p| std::fs::read(p).ok());
    let key_bytes = key_path.and_then(|p| std::fs::read(p).ok());
    add_tls(builder, &ca_bytes, cert_bytes, key_bytes)
}

/// Configure TLS on a `reqwest::ClientBuilder` using a hyperactor
/// [`PemBundle`](hyperactor::config::PemBundle).
///
/// The bundle provides a CA (`bundle.ca`) and optionally a client
/// certificate and key (`bundle.cert`, `bundle.key`). This reads the
/// CA via [`read_pem`] and installs it as a root trust anchor. If the
/// CA is missing, unreadable, or empty, returns `(builder, false)`
/// with no changes.
///
/// If both client cert and key are readable, they are combined and
/// passed to [`add_tls`] to configure an mTLS identity; otherwise the
/// client identity is omitted (CA-only TLS verification still
/// applies).
///
/// Returns `(updated_builder, ca_installed)`.
fn add_tls_from_bundle(
    builder: reqwest::ClientBuilder,
    bundle: &hyperactor::config::PemBundle,
) -> (reqwest::ClientBuilder, bool) {
    let ca_bytes = match read_pem(&bundle.ca) {
        Some(b) => b,
        None => {
            eprintln!("TLS: CA not readable from PemBundle");
            return (builder, false);
        }
    };
    add_tls(
        builder,
        &ca_bytes,
        read_pem(&bundle.cert),
        read_pem(&bundle.key),
    )
}

/// Build a `reqwest` client and a base URL from CLI arguments.
///
/// `args.addr` may be either a bare `host:port` or an explicit URL
/// (`http://host:port` / `https://host:port`). If an explicit scheme
/// is provided, that scheme is honored.
///
/// TLS configuration is applied in priority order:
/// 0. If `config.plaintext` is set, disable TLS and use plain HTTP.
/// 1. If `--tls-ca` is provided, attempt to load the CA (and
///    optionally `--tls-cert` + `--tls-key` for a client identity)
///    from those paths.
/// 2. Otherwise, if no `--tls-ca` was given, try auto-detection via
///    [`hyperactor::channel::try_tls_pem_bundle`] (OSS config first,
///    then Meta well-known paths). This runs even when the user
///    provides an explicit `https://` scheme, so the mTLS client
///    identity is picked up from well-known paths.
/// 3. If no CA can be loaded, fall back to plain HTTP.
///
/// Returns `(base_url, client)` where `base_url` always includes the
/// scheme selected (`http://...` or `https://...`).
pub(crate) fn build_client(config: &TuiConfig) -> (String, reqwest::Client) {
    let (explicit_scheme, host) = parse_addr(&config.addr);

    // TP-7: no client-level timeout. All timeout enforcement is
    // per-operation via tokio::time::timeout at the call boundary.
    let mut builder = reqwest::Client::builder();
    let mut use_tls = !config.plaintext && explicit_scheme == Some("https");

    // 1. Explicit CLI cert paths (skipped when --plaintext).
    if !config.plaintext
        && let Some(ca_path) = &config.tls_ca
    {
        let (b, ok) = add_tls_from_paths(
            builder,
            ca_path,
            config.tls_cert.as_deref(),
            config.tls_key.as_deref(),
        );
        builder = b;
        use_tls = use_tls || ok;
    }

    // 2. Auto-detect (when no CLI certs were provided).
    // This runs even with an explicit https:// scheme, so the client
    // picks up the mTLS identity from Meta well-known paths.
    if !config.plaintext
        && config.tls_ca.is_none()
        && let Some(bundle) = hyperactor::channel::try_tls_pem_bundle()
    {
        let (b, ok) = add_tls_from_bundle(builder, &bundle);
        builder = b;
        use_tls = use_tls || ok;
    }

    let scheme = if use_tls { "https" } else { "http" };
    let base_url = format!("{}://{}", scheme, host);
    let client = builder.build().unwrap_or_else(|_| reqwest::Client::new());

    (base_url, client)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A valid self-signed CA PEM, copied from hyperactor's test suite.
    /// `reqwest::Certificate::from_pem` parses it regardless of expiry, so
    /// it exercises the real "CA installs, TLS enabled" path.
    const TEST_CA_CERT: &str = "-----BEGIN CERTIFICATE-----
MIIDBTCCAe2gAwIBAgIUaGNmboiIosG+8Up0vgDr/+cg+2IwDQYJKoZIhvcNAQEL
BQAwEjEQMA4GA1UEAwwHVGVzdCBDQTAeFw0yNjAxMjgxNzA4MzlaFw0yNzAxMjgx
NzA4MzlaMBIxEDAOBgNVBAMMB1Rlc3QgQ0EwggEiMA0GCSqGSIb3DQEBAQUAA4IB
DwAwggEKAoIBAQC9RBoMYXCajklswt8Vi1JI1lEYzic0WNOmz45vG/7H6jTWkgL3
K5Ri+Seg3MobDNc48YHWXYm4hP9wCzkx8ih3ntT5XiY1My/G3jLUuoIEE9pF/BoJ
YQwZVoPNFhA9WhXNRsINf1cXFf8NzRfXpxBfKWtQJxYXU4JiDBQ6rLnQQABo8JmQ
vYFhJbBaYip5jTSiVNn7mB1zNr5jsVxuoSF53Pb7xQ76bwBdOq4zd6PSxL5/lr4G
cHSoxwZQdZMG7PL6hbxDQ2S2YI2lYVET1zwc2WPKCfjbEXBC/jzx828CInQtuksk
18gJt6xHkTFEA8CSA29GM3lejnwYWf51xyyBAgMBAAGjUzBRMB0GA1UdDgQWBBRX
cbxSZ70NsUkAS3Hhy6irugywJDAfBgNVHSMEGDAWgBRXcbxSZ70NsUkAS3Hhy6ir
ugywJDAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQA7aAFfyW67
Z+uGSVYhpsT/uH/3Z3nr7X1smTz5CGEfq2czEcTC7gbYI2l8GZ47GPfnAvHTBZVm
V/XncBCsj7/thOh2jYEHFyCbPckoaSCRyCOnK7LPUlr4HN5uP9EFe45qBLCJDEoY
GTTw7MtzwdovfjchNfKQCTtkBJCXQ95WLCf6UOh02Sn28UTlgfXzF0X0FrcWqWa3
uJZd4XOo4O6hKKlHaBaQPiEr++1xc3SWPV7jZHbckI/vKBnDdEZ9JQX5fFZuypUI
sgomYHxvxrU2hWx+7k53CRdjfaIvT9Ie44z9sSdsU/+blw2S8f/ZTmuECoIAAXYO
0qpzlxZMdr7T
-----END CERTIFICATE-----
";

    fn config(addr: &str, plaintext: bool, tls_ca: Option<String>) -> TuiConfig {
        TuiConfig {
            addr: addr.to_string(),
            refresh_ms: 2000,
            theme: crate::ThemeName::Nord,
            lang: crate::theme::LangName::En,
            tls_ca,
            tls_cert: None,
            tls_key: None,
            diagnose: false,
            plaintext,
        }
    }

    #[test]
    fn plaintext_forces_http_over_https_scheme() {
        // upholds: TUI-T1 -- --plaintext is the top-priority transport override.
        let (base_url, _) = build_client(&config("https://host:1729", true, None));
        assert!(
            base_url.starts_with("http://"),
            "plaintext must force plain HTTP even for an https:// addr, got {base_url}"
        );
    }

    #[test]
    fn without_plaintext_https_scheme_stays_https() {
        // negative: default posture is unchanged when the flag is off.
        let (base_url, _) = build_client(&config("https://host:1729", false, None));
        assert!(
            base_url.starts_with("https://"),
            "https:// must stay TLS when --plaintext is unset, got {base_url}"
        );
    }

    #[test]
    fn plaintext_beats_a_valid_explicit_ca() {
        // A genuinely loadable CA sets use_tls without the guard, so a bogus
        // path would pass vacuously; the CA must actually parse to prove
        // --plaintext wins.
        let path =
            std::env::temp_dir().join(format!("tui_admin_test_ca_{}.pem", std::process::id()));
        std::fs::write(&path, TEST_CA_CERT).expect("write temp CA");
        let ca = path.to_string_lossy().into_owned();

        let (tls_url, _) = build_client(&config("host:1729", false, Some(ca.clone())));
        assert!(
            tls_url.starts_with("https://"),
            "a valid --tls-ca should enable TLS, got {tls_url}"
        );

        let (plain_url, _) = build_client(&config("host:1729", true, Some(ca)));
        assert!(
            plain_url.starts_with("http://"),
            "plaintext must win over a valid --tls-ca, got {plain_url}"
        );

        std::fs::remove_file(&path).ok();
    }
}
