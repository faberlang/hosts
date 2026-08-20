use super::*;
use host_kernel::ProviderContent;

fn context() -> DispatchContext {
    DispatchContext {
        cancellation: host_kernel::CancellationProbe::new(|| false),
    }
}

#[test]
fn manifest_contains_canonical_routes_and_omits_legacy_aliases() {
    let mut kernel = Kernel::new();
    register(&mut kernel).expect("register solum");
    let calls = &kernel.manifest().providers[0].calls;
    assert_eq!(calls.len(), 46);
    assert!(calls.iter().any(|call| call.route == "solum:modum"));
    assert!(calls.iter().any(|call| call.route == "solum:vincula"));
    assert!(calls.iter().any(|call| call.route == "solum:digestio"));
    assert!(!calls.iter().any(|call| call.route == "solum:fundet"));
    assert!(!calls.iter().any(|call| call.route == "solum:leget"));
}

#[test]
fn mode_and_relative_symlink_operations_preserve_contract() {
    let provider = Solum::new().expect("provider");
    let dir = std::env::temp_dir().join(format!("faber-public-solum-{}", std::process::id()));
    let file = dir.join("payload.txt");
    let link = dir.join("payload-link.txt");
    std::fs::create_dir(&dir).expect("fixture directory");
    std::fs::write(&file, "salve").expect("fixture file");

    let set_mode = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "mode".into(),
                route: "solum:modum".into(),
                opener: Valor::Lista(vec![
                    Valor::Textus(file.to_string_lossy().into_owned()),
                    Valor::Numerus(0o640),
                ]),
                target: None,
            },
            &context(),
        )
        .expect("set mode");
    assert!(set_mode.contents.is_empty());
    assert_eq!(
        std::fs::metadata(&file).expect("stat").permissions().mode() & 0o7777,
        0o640
    );

    let link_reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "link".into(),
                route: "solum:vincula".into(),
                opener: Valor::Lista(vec![
                    Valor::Textus("payload.txt".into()),
                    Valor::Textus(link.to_string_lossy().into_owned()),
                ]),
                target: None,
            },
            &context(),
        )
        .expect("symlink");
    assert!(link_reply.contents.is_empty());
    assert_eq!(
        std::fs::read_link(&link).expect("read link"),
        Path::new("payload.txt")
    );
    assert!(std::fs::symlink_metadata(&link)
        .expect("stat link")
        .file_type()
        .is_symlink());

    std::fs::remove_file(&link).expect("cleanup link");
    std::fs::remove_file(&file).expect("cleanup file");
    std::fs::remove_dir(&dir).expect("cleanup dir");
}

#[test]
fn bounded_partem_and_inveni_return_byte_and_scalar_shapes() {
    let provider = Solum::new().expect("provider");
    let path =
        std::env::temp_dir().join(format!("faber-public-solum-range-{}", std::process::id()));
    std::fs::write(&path, "salve munde").expect("fixture");
    let part = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "part".into(),
                route: "solum:partem".into(),
                opener: Valor::Lista(vec![
                    Valor::Textus(path.to_string_lossy().into_owned()),
                    Valor::Numerus(6),
                    Valor::Numerus(5),
                ]),
                target: None,
            },
            &context(),
        )
        .expect("part");
    assert!(
        matches!(part.contents.as_slice(), [ProviderContent::Byte(bytes)] if bytes == b"munde")
    );
    let found = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "find".into(),
                route: "solum:inveni".into(),
                opener: Valor::Lista(vec![
                    Valor::Textus(path.to_string_lossy().into_owned()),
                    Valor::Textus("munde".into()),
                    Valor::Numerus(0),
                    Valor::Numerus(32),
                ]),
                target: None,
            },
            &context(),
        )
        .expect("find");
    assert!(matches!(
        found.contents.as_slice(),
        [ProviderContent::Item(Valor::Numerus(6))]
    ));
    std::fs::remove_file(path).expect("cleanup");
}

#[test]
#[allow(clippy::cast_possible_wrap)]
fn partem_and_inveni_reject_over_limit_ranges_before_allocation() {
    let provider = Solum::new().expect("provider");
    let path =
        std::env::temp_dir().join(format!("faber-public-solum-limit-{}", std::process::id()));
    std::fs::write(&path, b"payload").expect("fixture");
    let path_s = path.to_string_lossy().into_owned();

    let zero_part = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "part-zero".into(),
                route: "solum:partem".into(),
                opener: Valor::Lista(vec![
                    Valor::Textus(path_s.clone()),
                    Valor::Numerus(0),
                    Valor::Numerus(0),
                ]),
                target: None,
            },
            &context(),
        )
        .expect("zero-length partem");
    assert!(
        matches!(zero_part.contents.as_slice(), [ProviderContent::Byte(bytes)] if bytes.is_empty())
    );

    for (route, opener) in [
        (
            "solum:partem",
            Valor::Lista(vec![
                Valor::Textus(path_s.clone()),
                Valor::Numerus(0),
                Valor::Numerus(MAX_RANGE_READ_BYTES as i64 + 1),
            ]),
        ),
        (
            "solum:inveni",
            Valor::Lista(vec![
                Valor::Textus(path_s.clone()),
                Valor::Textus("pay".into()),
                Valor::Numerus(0),
                Valor::Numerus(MAX_RANGE_READ_BYTES as i64 + 1),
            ]),
        ),
    ] {
        let error = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: format!("{route}-too-long"),
                    route: route.to_owned(),
                    opener,
                    target: None,
                },
                &context(),
            )
            .expect_err("over-limit range must fail before allocation");
        assert_eq!(error.code, "E_INVALID_ARGS");
        assert!(error.message.contains(route));
        assert!(error.message.contains(&MAX_RANGE_READ_BYTES.to_string()));
    }

    for (route, opener) in [
        (
            "solum:partem",
            Valor::Lista(vec![
                Valor::Textus(path_s.clone()),
                Valor::Numerus(0),
                Valor::Numerus(-1),
            ]),
        ),
        (
            "solum:inveni",
            Valor::Lista(vec![
                Valor::Textus(path_s),
                Valor::Textus("pay".into()),
                Valor::Numerus(0),
                Valor::Numerus(-1),
            ]),
        ),
    ] {
        let error = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: format!("{route}-negative"),
                    route: route.to_owned(),
                    opener,
                    target: None,
                },
                &context(),
            )
            .expect_err("negative range length must remain invalid");
        assert_eq!(error.code, "E_INVALID_ARGS");
        assert!(error.message.contains("longitudo"));
        assert!(error.message.contains("non-negative"));
    }

    std::fs::remove_file(path).expect("cleanup");
}

#[test]
fn lege_is_textus_only_and_rejects_list_or_byte_targets() {
    let provider = Solum::new().expect("provider");
    let path = std::env::temp_dir().join(format!("faber-public-solum-lege-{}", std::process::id()));
    std::fs::write(&path, "prima\nsecunda\n").expect("fixture");
    let path_s = path.to_string_lossy().into_owned();

    let text = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "lege-text".into(),
                route: "solum:lege".into(),
                opener: Valor::Textus(path_s.clone()),
                target: Some(std::any::type_name::<String>().to_owned()),
            },
            &context(),
        )
        .expect("lege text");
    assert!(matches!(
        text.contents.as_slice(),
        [ProviderContent::Item(Valor::Textus(s))] if s == "prima\nsecunda\n"
    ));

    for target in [
        std::any::type_name::<Vec<String>>(),
        std::any::type_name::<Vec<u8>>(),
    ] {
        let error = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "lege-target".into(),
                    route: "solum:lege".into(),
                    opener: Valor::Textus(path_s.clone()),
                    target: Some(target.to_owned()),
                },
                &context(),
            )
            .expect_err("non-text solum:lege target must not bypass manifest contract");
        assert_eq!(error.code, "E_INTERNAL");
        assert!(error.message.contains("solum:lege target"));
        assert!(error.message.contains("solum:carpe"));
        assert!(error.message.contains("solum:hauri"));
    }
    std::fs::remove_file(&path).expect("cleanup");
}

#[test]
fn inveni_empty_pattern_is_found_at_start() {
    let provider = Solum::new().expect("provider");
    let path =
        std::env::temp_dir().join(format!("faber-public-solum-empty-{}", std::process::id()));
    std::fs::write(&path, b"payload").expect("fixture");
    let found = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "empty".into(),
                route: "solum:inveni".into(),
                opener: Valor::Lista(vec![
                    Valor::Textus(path.to_string_lossy().into_owned()),
                    Valor::Textus(String::new()),
                    Valor::Numerus(3),
                    Valor::Numerus(8),
                ]),
                target: None,
            },
            &context(),
        )
        .expect("empty inveni");
    assert!(matches!(
        found.contents.as_slice(),
        [ProviderContent::Item(Valor::Numerus(3))]
    ));
    std::fs::remove_file(&path).expect("cleanup");
}

#[test]
fn dele_missing_path_is_success() {
    let provider = Solum::new().expect("provider");
    let missing =
        std::env::temp_dir().join(format!("faber-public-solum-missing-{}", std::process::id()));
    assert!(!missing.exists());
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "dele-missing".into(),
                route: "solum:dele".into(),
                opener: Valor::Textus(missing.to_string_lossy().into_owned()),
                target: None,
            },
            &context(),
        )
        .expect("dele missing");
    assert!(reply.contents.is_empty());
}

#[test]
fn tange_existing_socket_returns_internal_error_instead_of_success() {
    use std::os::unix::net::UnixListener;

    let provider = Solum::new().expect("provider");
    let path =
        std::env::temp_dir().join(format!("faber-public-solum-socket-{}", std::process::id()));
    let listener = UnixListener::bind(&path).expect("bind socket fixture");
    let path_s = path.to_string_lossy().into_owned();
    let error = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "tange-socket".into(),
                route: "solum:tange".into(),
                opener: Valor::Textus(path_s.clone()),
                target: None,
            },
            &context(),
        )
        .expect_err("touching an unopenable existing path must fail");
    assert_eq!(error.code, "E_INTERNAL");
    assert!(error.message.contains("solum:tange"));
    assert!(error.message.contains(&path_s));

    drop(listener);
    let _ = std::fs::remove_file(path);
}

#[test]
fn home_value_prefers_home_then_userprofile_and_errors_without_either() {
    assert_eq!(
        home_value(Some("/home/faber".into()), Some("C:\\Users\\faber".into())),
        Ok("/home/faber".into())
    );
    assert_eq!(
        home_value(None, Some("C:\\Users\\faber".into())),
        Ok("C:\\Users\\faber".into())
    );
    assert_eq!(
        home_value(None, None),
        Err("no home directory environment variable")
    );
}

// FIPS 180-4 SHA-256("abc") — the pinned file-digest oracle.
const FIPS_ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

#[test]
fn digestio_of_known_file_matches_fips_180_4_abc() {
    let provider = Solum::new().expect("provider");
    let path = std::env::temp_dir().join(format!(
        "faber-public-solum-digestio-{}",
        std::process::id()
    ));
    std::fs::write(&path, b"abc").expect("fixture");
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "digestio-abc".into(),
                route: "solum:digestio".into(),
                opener: Valor::Textus(path.to_string_lossy().into_owned()),
                target: None,
            },
            &context(),
        )
        .expect("solum:digestio");
    assert!(
        matches!(
            reply.contents.as_slice(),
            [ProviderContent::Item(Valor::Textus(hex))] if hex == FIPS_ABC_SHA256
        ),
        "solum:digestio must return the pinned SHA-256 hex, got {reply:?}"
    );
    std::fs::remove_file(path).expect("cleanup");
}

#[test]
fn digestio_of_empty_file_matches_fips_180_4_empty() {
    let provider = Solum::new().expect("provider");
    let path = std::env::temp_dir().join(format!(
        "faber-public-solum-digestio-empty-{}",
        std::process::id()
    ));
    std::fs::write(&path, b"").expect("fixture");
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "digestio-empty".into(),
                route: "solum:digestio".into(),
                opener: Valor::Textus(path.to_string_lossy().into_owned()),
                target: None,
            },
            &context(),
        )
        .expect("solum:digestio empty");
    assert!(matches!(
        reply.contents.as_slice(),
        [ProviderContent::Item(Valor::Textus(hex))]
            if hex == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ));
    std::fs::remove_file(path).expect("cleanup");
}

#[test]
fn digestio_missing_path_is_internal_error() {
    let provider = Solum::new().expect("provider");
    let missing = std::env::temp_dir().join(format!(
        "faber-public-solum-digestio-missing-{}",
        std::process::id()
    ));
    assert!(!missing.exists());
    let error = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "digestio-missing".into(),
                route: "solum:digestio".into(),
                opener: Valor::Textus(missing.to_string_lossy().into_owned()),
                target: None,
            },
            &context(),
        )
        .expect_err("missing file must fail");
    assert_eq!(error.code, "E_INTERNAL");
    assert!(error.message.contains("solum:digestio"));
}

#[test]
fn exstat_nonexistent_path_returns_false() {
    let provider = Solum::new().expect("provider");
    let dir = tempfile::tempdir().expect("temp dir");
    let missing = dir.path().join("solum-nonexistent");
    assert!(!missing.exists());
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "exstat-missing".into(),
                route: "solum:exstat".into(),
                opener: Valor::Textus(missing.to_string_lossy().into_owned()),
                target: None,
            },
            &context(),
        )
        .expect("exstat missing");
    assert!(matches!(
        reply.contents.as_slice(),
        [ProviderContent::Item(Valor::Bivalens(false))]
    ));
}

#[test]
fn crea_existing_directory_is_idempotent() {
    let provider = Solum::new().expect("provider");
    let dir = tempfile::tempdir().expect("temp dir");
    let existing = dir.path().join("solum-crea-exist");
    std::fs::create_dir(&existing).expect("first create");
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "crea-existing".into(),
                route: "solum:crea".into(),
                opener: Valor::Textus(existing.to_string_lossy().into_owned()),
                target: None,
            },
            &context(),
        )
        .expect("crea existing");
    assert!(reply.contents.is_empty());
    assert!(existing.exists());
}

#[test]
fn regula_rejects_non_existent_path() {
    let provider = Solum::new().expect("provider");
    let dir = tempfile::tempdir().expect("temp dir");
    let missing = dir.path().join("solum-regula-missing");
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "regula-missing".into(),
                route: "solum:regularene".into(),
                opener: Valor::Textus(missing.to_string_lossy().into_owned()),
                target: None,
            },
            &context(),
        )
        .expect("regula missing");
    assert!(matches!(
        reply.contents.as_slice(),
        [ProviderContent::Item(Valor::Bivalens(false))]
    ));
}
