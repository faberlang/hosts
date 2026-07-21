use std::collections::BTreeSet;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use faber::{frame, FrameStatus, Instans, InstansPraecisio, Scrinium, Valor};
use faber_host_macos_arm64::kernel::frame_data;
use faber_host_macos_arm64::kernel::valor_wire::valor_to_json;
use faber_host_macos_arm64::{Direction, Frame, HostKernel, Status};
use serde_json::Value;

fn data_json(data: &Valor) -> Value {
    valor_to_json(data).expect("frame data should encode to JSON")
}

fn temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("faber-host-{name}-{}-{nanos}", std::process::id()))
}

#[test]
fn routes_host_echo_as_done_frame() {
    let kernel = HostKernel::new();
    let data = frame_data::tabula([("value", Valor::Textus("salve".into()))]);
    let request = Frame::request_with("host:echo", data);

    let response = kernel.route(&request);

    assert_eq!(response.status, Status::Done);
    assert_eq!(response.parent_id.as_deref(), Some(request.id.as_str()));
    assert_eq!(response.call, "host:echo");
    assert_eq!(
        data_json(&response.data)["echo"]["value"],
        Value::String("salve".into())
    );
}

#[test]
fn opens_host_echo_as_bidirectional_conversation() {
    let kernel = HostKernel::new();
    let data = frame_data::tabula([("value", Valor::Textus("salve".into()))]);
    let request = Frame::request_with("host:echo", data);
    let request_id = request.id.clone();

    let mut conversation = kernel.open(request);
    conversation.push(Frame::request_with(
        "ignored:content",
        frame_data::tabula([("chunk", Valor::Textus("caller".into()))]),
    ));
    conversation.done(Direction::CallerToGateway);

    let response = conversation.recv().expect("gateway item response");
    assert_eq!(conversation.id(), request_id);
    assert_eq!(conversation.route(), "host:echo");
    assert_eq!(response.status, Status::Item);
    assert_eq!(response.parent_id.as_deref(), Some(request_id.as_str()));
    assert_eq!(
        data_json(&response.data)["echo"]["value"],
        Value::String("salve".into())
    );
    let done = conversation.recv().expect("gateway terminal response");
    assert_eq!(done.status, Status::Done);
    assert_eq!(done.parent_id.as_deref(), Some(request_id.as_str()));
    assert!(conversation.recv().is_none());
    assert!(conversation.is_complete());
    assert!(conversation
        .sent()
        .iter()
        .any(|frame| frame.status == Status::Done));
}

#[test]
fn attaches_runtime_sermo_to_host_echo_conversation() {
    let kernel = HostKernel::new();
    let mut sermo = frame::sermo_open("host:echo");
    frame::sermo_set_opener(
        &mut sermo,
        frame_data::tabula([("value", Valor::Textus("salve".into()))]),
    );

    kernel
        .attach_sermo(&mut sermo)
        .expect("host echo sermo attaches");

    let item = frame::sermo_recv(&mut sermo).expect("host echo item frame");
    assert_eq!(item.status, FrameStatus::Item);
    assert_eq!(
        item.parent_id.as_deref(),
        Some(sermo.conversation_id().as_str())
    );
    assert_eq!(item.call, "host:echo");
    assert_eq!(
        data_json(&item.data)["echo"]["value"],
        Value::String("salve".into())
    );

    let done = frame::sermo_recv(&mut sermo).expect("host echo done frame");
    assert_eq!(done.status, FrameStatus::Done);
    assert!(sermo.incoming_drained());
    assert!(frame::sermo_recv(&mut sermo).is_none());
}

#[test]
fn routes_host_bytes_as_byte_status_payload() {
    let kernel = HostKernel::new();
    let request = Frame::request("host:bytes");

    let response = kernel.route(&request);

    assert_eq!(response.status, Status::Byte);
    assert_eq!(response.parent_id.as_deref(), Some(request.id.as_str()));
    assert_eq!(
        response.data,
        Valor::Lista(vec![1_i64.into(), 2_i64.into(), 3_i64.into()])
    );
}

#[test]
fn opens_host_bytes_with_terminal_after_byte_frame() {
    let kernel = HostKernel::new();
    let request = Frame::request("host:bytes");
    let request_id = request.id.clone();

    let mut conversation = kernel.open(request);
    conversation.done(Direction::CallerToGateway);

    let bytes = conversation.recv().expect("gateway byte response");
    assert_eq!(bytes.status, Status::Byte);
    assert_eq!(bytes.parent_id.as_deref(), Some(request_id.as_str()));
    assert_eq!(
        bytes.data,
        Valor::Lista(vec![1_i64.into(), 2_i64.into(), 3_i64.into()])
    );

    let done = conversation.recv().expect("gateway terminal response");
    assert_eq!(done.status, Status::Done);
    assert_eq!(done.parent_id.as_deref(), Some(request_id.as_str()));
    assert!(conversation.recv().is_none());
    assert!(conversation.is_complete());
}

#[test]
fn converts_runtime_scrinium_into_host_frame() {
    let frame = Frame::from(Scrinium {
        id: "runtime-frame".into(),
        parent_id: Some("root".into()),
        call: "host:bytes".into(),
        status: FrameStatus::Byte,
        data: Valor::Lista(vec![9_i64.into()]),
        created_ms: 42,
        from: Some("generated-rust".into()),
        trace: Some(Valor::Textus("debug".into())),
    });

    assert_eq!(frame.id, "runtime-frame");
    assert_eq!(frame.parent_id.as_deref(), Some("root"));
    assert_eq!(frame.call, "host:bytes");
    assert_eq!(frame.status, Status::Byte);
    assert_eq!(frame.created_ms, 42);
    assert_eq!(frame.from.as_deref(), Some("generated-rust"));
    assert_eq!(frame.trace, Some(Value::String("debug".into())));
    assert_eq!(frame.data, Valor::Lista(vec![9_i64.into()]));
}

#[test]
fn converts_host_frame_trace_back_to_runtime_scrinium() {
    let frame = Frame::request("host:echo").with_trace(Value::String("debug".into()));

    let scrinium = Scrinium::from(frame);

    assert_eq!(scrinium.trace, Some(Valor::Textus("debug".into())));
}

#[test]
fn reports_unresolved_call_as_no_route_error_frame() {
    let kernel = HostKernel::new();
    let request = Frame::request("pg:query");

    let response = kernel.route(&request);

    assert_eq!(response.status, Status::Error);
    assert_eq!(response.parent_id.as_deref(), Some(request.id.as_str()));
    assert_eq!(
        data_json(&response.data)["code"],
        Value::String("E_NO_ROUTE".into())
    );
    assert_eq!(data_json(&response.data)["retryable"], Value::Bool(false));
}

#[test]
fn manifest_lists_builtin_host_echo_and_no_default_providers() {
    let kernel = HostKernel::new();

    let manifest = kernel.manifest();

    assert_eq!(manifest.host, "macos-arm64");
    assert_eq!(manifest.manifest_version, 1);
    assert!(manifest
        .builtins
        .iter()
        .any(|item| item.name == "host:echo"));
    assert!(manifest
        .builtins
        .iter()
        .any(|item| item.name == "consolum:scribe"));
    assert!(manifest.providers.is_empty());
}

#[test]
fn manifest_lists_current_stdlib_sync_routes() {
    let kernel = HostKernel::new();
    let manifest = kernel.manifest();
    let builtin_names: Vec<&str> = manifest
        .builtins
        .iter()
        .map(|item| item.name.as_str())
        .collect();

    for expected in [
        "consolum:hauri",
        "consolum:lege",
        "consolum:scribe",
        "consolum:dic",
        "consolum:mone",
        "consolum:vide",
        "consolum:audit",
        "consolum:loquitur",
        "consolum:admonet",
        "solum:lege",
        "solum:carpe",
        "solum:scribe",
        "solum:appone",
        "solum:partem",
        "solum:inveni",
        "solum:exstat",
        "solum:modum",
        "solum:vincula",
        "solum:dele",
        "solum:exscribe",
        "solum:renomina",
        "solum:crea",
        "solum:enumera",
        "solum:domus",
        "solum:temporarium",
        "processus:exsequi",
        "processus:lege",
        "processus:scribe",
        "processus:sedes",
        "processus:muta",
        "processus:identitas",
        "processus:argumenta",
        "processus:captura",
        "aleator:fractum",
        "aleator:sortire",
        "aleator:octetos",
        "aleator:uuid",
        "aleator:semina",
        "tempus:nunc",
        "tempus:monotonicum",
        "tempus:activum",
    ] {
        assert!(builtin_names.contains(&expected), "missing {expected}");
    }
}

#[test]
fn routes_consolum_stderr_output_as_done_frame() {
    let kernel = HostKernel::new();
    let data = Valor::Textus(String::new());
    let request = Frame::request_with("consolum:vide", data);

    let response = kernel.route(&request);

    assert_eq!(response.status, Status::Done);
    assert_eq!(response.parent_id.as_deref(), Some(request.id.as_str()));
    assert_eq!(response.call, "consolum:vide");
}

#[test]
fn routes_consolum_tty_predicate_as_boolean_frame_data() {
    let kernel = HostKernel::new();
    let request = Frame::request("consolum:loquitur");

    let response = kernel.route(&request);

    assert_eq!(response.status, Status::Done);
    assert_eq!(response.parent_id.as_deref(), Some(request.id.as_str()));
    assert!(data_json(&response.data).is_boolean());
}

#[test]
fn rejects_consolum_tabula_opener_for_line_writes() {
    // Public consolum takes a string (or list string) opener, not a legacy
    // `{ "msg": ... }` tabula debug shape.
    let kernel = HostKernel::new();
    let data = frame_data::tabula([("msg", Valor::Textus(String::new()))]);
    let request = Frame::request_with("consolum:vide", data);

    let response = kernel.route(&request);

    assert_eq!(response.status, Status::Error);
    assert_eq!(
        data_json(&response.data)["code"],
        Value::String("E_INVALID_ARGS".into())
    );
}

#[test]
fn reports_consolum_bad_payload_as_invalid_args() {
    let kernel = HostKernel::new();
    let request = Frame::request("consolum:scribe");

    let response = kernel.route(&request);

    assert_eq!(response.status, Status::Error);
    assert_eq!(response.parent_id.as_deref(), Some(request.id.as_str()));
    assert_eq!(
        data_json(&response.data)["code"],
        Value::String("E_INVALID_ARGS".into())
    );
    assert_eq!(data_json(&response.data)["retryable"], Value::Bool(false));
}

#[test]
fn reports_consolum_missing_required_read_size_as_invalid_args() {
    let kernel = HostKernel::new();
    let request = Frame::request("consolum:hauri");

    let response = kernel.route(&request);

    assert_eq!(response.status, Status::Error);
    assert_eq!(response.parent_id.as_deref(), Some(request.id.as_str()));
    assert_eq!(
        data_json(&response.data)["code"],
        Value::String("E_INVALID_ARGS".into())
    );
    let payload = data_json(&response.data);
    let message = payload["message"].as_str().expect("message");
    assert!(
        message.contains("magnitudo"),
        "expected magnitudo validation message, got {message}"
    );
}

#[test]
fn reports_unknown_consolum_member_as_no_route() {
    let kernel = HostKernel::new();
    let request = Frame::request("consolum:ignotum");

    let response = kernel.route(&request);

    assert_eq!(response.status, Status::Error);
    assert_eq!(response.parent_id.as_deref(), Some(request.id.as_str()));
    assert_eq!(
        data_json(&response.data)["code"],
        Value::String("E_NO_ROUTE".into())
    );
}

#[test]
fn routes_solum_text_file_operations_from_ordered_payloads() {
    let kernel = HostKernel::new();
    let dir = temp_path("solum-dir");
    let file = dir.join("note.txt");
    let copy = dir.join("copy.txt");
    let moved = dir.join("moved.txt");
    let link = dir.join("note-link.txt");

    let create = kernel.route(&Frame::request_with(
        "solum:crea",
        Valor::Textus(dir.to_string_lossy().into_owned()),
    ));
    assert_eq!(create.status, Status::Done);

    let write = kernel.route(&Frame::request_with(
        "solum:scribe",
        Valor::Lista(vec![
            Valor::Textus(file.to_string_lossy().into_owned()),
            Valor::Textus("salve".into()),
        ]),
    ));
    assert_eq!(write.status, Status::Done);

    let append = kernel.route(&Frame::request_with(
        "solum:appone",
        Valor::Lista(vec![
            Valor::Textus(file.to_string_lossy().into_owned()),
            Valor::Textus("\nmunde".into()),
        ]),
    ));
    assert_eq!(append.status, Status::Done);

    let read = kernel.route(&Frame::request_with(
        "solum:lege",
        Valor::Textus(file.to_string_lossy().into_owned()),
    ));
    assert_eq!(read.status, Status::Done);
    assert_eq!(read.data, Valor::Textus("salve\nmunde".into()));

    let part = kernel.route(&Frame::request_with(
        "solum:partem",
        Valor::Lista(vec![
            Valor::Textus(file.to_string_lossy().into_owned()),
            Valor::Numerus(6),
            Valor::Numerus(5),
        ]),
    ));
    // Public solum returns byte-status frames for ranged binary reads.
    assert_eq!(part.status, Status::Byte);
    assert_eq!(part.data, Valor::Octeti(b"munde".to_vec()));

    let found = kernel.route(&Frame::request_with(
        "solum:inveni",
        Valor::Lista(vec![
            Valor::Textus(file.to_string_lossy().into_owned()),
            Valor::Textus("munde".into()),
            Valor::Numerus(0),
            Valor::Numerus(32),
        ]),
    ));
    assert_eq!(found.status, Status::Done);
    assert_eq!(found.data, Valor::Numerus(6));

    let exists = kernel.route(&Frame::request_with(
        "solum:exstat",
        Valor::Textus(file.to_string_lossy().into_owned()),
    ));
    assert_eq!(exists.data, Valor::Bivalens(true));

    let regular = kernel.route(&Frame::request_with(
        "solum:regularene",
        Valor::Textus(file.to_string_lossy().into_owned()),
    ));
    assert_eq!(regular.data, Valor::Bivalens(true));

    let readable = kernel.route(&Frame::request_with(
        "solum:legibilene",
        Valor::Textus(file.to_string_lossy().into_owned()),
    ));
    assert_eq!(readable.data, Valor::Bivalens(true));

    let directory = kernel.route(&Frame::request_with(
        "solum:directoriumne",
        Valor::Textus(dir.to_string_lossy().into_owned()),
    ));
    assert_eq!(directory.data, Valor::Bivalens(true));

    let size = kernel.route(&Frame::request_with(
        "solum:mensura",
        Valor::Textus(file.to_string_lossy().into_owned()),
    ));
    assert_eq!(size.data, Valor::Numerus(11));

    let mode = kernel.route(&Frame::request_with(
        "solum:modus",
        Valor::Textus(file.to_string_lossy().into_owned()),
    ));
    assert!(matches!(mode.data, Valor::Numerus(bits) if bits & 0o400 != 0));

    std::os::unix::fs::symlink(&file, &link).expect("create symlink fixture");
    let symlink = kernel.route(&Frame::request_with(
        "solum:vinculumne",
        Valor::Textus(link.to_string_lossy().into_owned()),
    ));
    assert_eq!(symlink.data, Valor::Bivalens(true));

    let copy_response = kernel.route(&Frame::request_with(
        "solum:exscribe",
        Valor::Lista(vec![
            Valor::Textus(file.to_string_lossy().into_owned()),
            Valor::Textus(copy.to_string_lossy().into_owned()),
        ]),
    ));
    assert_eq!(copy_response.status, Status::Done);

    let move_response = kernel.route(&Frame::request_with(
        "solum:renomina",
        Valor::Lista(vec![
            Valor::Textus(copy.to_string_lossy().into_owned()),
            Valor::Textus(moved.to_string_lossy().into_owned()),
        ]),
    ));
    assert_eq!(move_response.status, Status::Done);

    let listing = kernel.route(&Frame::request_with(
        "solum:enumera",
        Valor::Textus(dir.to_string_lossy().into_owned()),
    ));
    assert_eq!(listing.status, Status::Done);
    assert_eq!(
        listing.data,
        Valor::Lista(vec![
            Valor::Textus("moved.txt".into()),
            Valor::Textus("note-link.txt".into()),
            Valor::Textus("note.txt".into())
        ])
    );

    let lines = kernel.open(Frame::request_with(
        "solum:carpe",
        Valor::Textus(file.to_string_lossy().into_owned()),
    ));
    let mut lines = lines;
    assert_eq!(
        lines.recv().expect("first line").data,
        Valor::Textus("salve".into())
    );
    assert_eq!(
        lines.recv().expect("second line").data,
        Valor::Textus("munde".into())
    );
    assert_eq!(lines.recv().expect("terminal").status, Status::Done);

    let delete = kernel.route(&Frame::request_with(
        "solum:dele",
        Valor::Textus(file.to_string_lossy().into_owned()),
    ));
    assert_eq!(delete.status, Status::Done);

    std::fs::remove_file(link).expect("cleanup symlink");
    std::fs::remove_file(moved).expect("cleanup moved file");
    std::fs::remove_dir(dir).expect("cleanup solum dir");
}

#[test]
fn routes_solum_mode_and_symlink_operations() {
    let kernel = HostKernel::new();
    let dir = temp_path("solum-mode-link");
    let file = dir.join("payload.txt");
    let link = dir.join("payload-link.txt");
    let relative_target = "payload.txt";

    std::fs::create_dir(&dir).expect("create solum fixture directory");
    std::fs::write(&file, "salve").expect("write solum fixture file");

    let set_mode = kernel.route(&Frame::request_with(
        "solum:modum",
        Valor::Lista(vec![
            Valor::Textus(file.to_string_lossy().into_owned()),
            Valor::Numerus(0o640),
        ]),
    ));
    assert_eq!(set_mode.status, Status::Done);
    let mode = std::fs::metadata(&file)
        .expect("stat mode fixture")
        .permissions()
        .mode();
    assert_eq!(mode & 0o7777, 0o640);

    for invalid_mode in [-1, 0o10000] {
        let response = kernel.route(&Frame::request_with(
            "solum:modum",
            Valor::Lista(vec![
                Valor::Textus(file.to_string_lossy().into_owned()),
                Valor::Numerus(invalid_mode),
            ]),
        ));
        assert_eq!(response.status, Status::Error);
        assert_eq!(
            data_json(&response.data)["code"],
            Value::String("E_INVALID_ARGS".into())
        );
    }

    let create_link = kernel.route(&Frame::request_with(
        "solum:vincula",
        Valor::Lista(vec![
            Valor::Textus(relative_target.into()),
            Valor::Textus(link.to_string_lossy().into_owned()),
        ]),
    ));
    assert_eq!(create_link.status, Status::Done);
    assert_eq!(
        std::fs::read_link(&link).expect("read symlink target"),
        PathBuf::from(relative_target)
    );
    assert!(std::fs::symlink_metadata(&link)
        .expect("stat symlink")
        .file_type()
        .is_symlink());

    let duplicate = kernel.route(&Frame::request_with(
        "solum:vincula",
        Valor::Lista(vec![
            Valor::Textus("different-target".into()),
            Valor::Textus(link.to_string_lossy().into_owned()),
        ]),
    ));
    assert_eq!(duplicate.status, Status::Error);

    std::fs::remove_file(&link).expect("cleanup symlink");
    std::fs::remove_file(&file).expect("cleanup mode fixture");
    std::fs::remove_dir(&dir).expect("cleanup solum mode directory");
}

#[test]
fn attaches_sermo_to_solum_lege_conversation() {
    let kernel = HostKernel::new();
    let file = temp_path("sermo-solum.txt");
    std::fs::write(&file, "salve").expect("write fixture");

    let mut sermo = frame::sermo_open("solum:lege");
    frame::sermo_set_opener(
        &mut sermo,
        Valor::Textus(file.to_string_lossy().into_owned()),
    );

    kernel.attach_sermo(&mut sermo).expect("attach solum sermo");

    let item = frame::sermo_recv(&mut sermo).expect("solum content frame");
    assert_eq!(item.status, FrameStatus::Item);
    assert_eq!(item.data, Valor::Textus("salve".into()));
    assert_eq!(
        frame::sermo_recv(&mut sermo).expect("terminal").status,
        FrameStatus::Done
    );

    std::fs::remove_file(file).expect("cleanup solum fixture");
}

#[test]
fn routes_processus_environment_and_identity() {
    let kernel = HostKernel::new();
    let name = format!("FABER_HOST_TEST_{}", std::process::id());

    let write = kernel.route(&Frame::request_with(
        "processus:scribe",
        Valor::Lista(vec![
            Valor::Textus(name.clone()),
            Valor::Textus("salve".into()),
        ]),
    ));
    assert_eq!(write.status, Status::Done);

    let read = kernel.route(&Frame::request_with(
        "processus:lege",
        Valor::Textus(name.clone()),
    ));
    assert_eq!(read.data, Valor::Textus("salve".into()));

    let sedes = kernel.route(&Frame::request("processus:sedes"));
    assert_eq!(sedes.status, Status::Done);
    assert!(matches!(sedes.data, Valor::Textus(_)));

    let identitas = kernel.route(&Frame::request("processus:identitas"));
    assert_eq!(identitas.data, Valor::Numerus(std::process::id() as i64));

    let args = kernel.route(&Frame::request("processus:argumenta"));
    assert_eq!(args.status, Status::Done);
    // Public processus returns the real process argv (skipping argv0). Empty
    // argv becomes a vacuum done frame; one arg is a single Textus item; many
    // args collapse to a Lista for the synchronous route() surface.
    match &args.data {
        Valor::Lista(items) => {
            assert!(items.iter().all(|item| matches!(item, Valor::Textus(_))));
        }
        Valor::Textus(_) | Valor::Nihil => {}
        Valor::Tabula(map) if map.is_empty() => {}
        other => panic!("unexpected argumenta payload: {other:?}"),
    }

    #[allow(unused_unsafe)]
    unsafe {
        std::env::remove_var(name);
    }
}

#[test]
fn routes_processus_shell_execution() {
    let kernel = HostKernel::new();
    let response = kernel.route(&Frame::request_with(
        "processus:exsequi",
        Valor::Textus("printf salve".into()),
    ));

    assert_eq!(response.status, Status::Done);
    assert_eq!(response.data, Valor::Textus("salve".into()));
}

#[test]
fn routes_processus_capture_accepts_root_text_list_opener() {
    let kernel = HostKernel::new();
    let response = kernel.route(&Frame::request_with(
        "processus:captura",
        Valor::Lista(vec![
            Valor::Textus("sh".into()),
            Valor::Textus("-c".into()),
            Valor::Textus("printf out; printf err >&2; exit 7".into()),
        ]),
    ));

    assert_eq!(response.status, Status::Done);
    let Valor::Tabula(fields) = response.data else {
        panic!("processus:captura must return tabula data");
    };
    assert_eq!(fields.get("status"), Some(&Valor::Numerus(7)));
    assert_eq!(fields.get("stdout"), Some(&Valor::Textus("out".into())));
    assert_eq!(fields.get("stderr"), Some(&Valor::Textus("err".into())));
}

#[test]
fn routes_aleator_seeded_and_crypto_values() {
    let kernel = HostKernel::new();

    assert_eq!(
        kernel
            .route(&Frame::request_with("aleator:semina", Valor::Numerus(42)))
            .status,
        Status::Done
    );
    let first = kernel.route(&Frame::request("aleator:fractum")).data;
    assert_eq!(
        kernel
            .route(&Frame::request_with("aleator:semina", Valor::Numerus(42)))
            .status,
        Status::Done
    );
    let second = kernel.route(&Frame::request("aleator:fractum")).data;
    assert_eq!(first, second);

    let sorted = kernel.route(&Frame::request_with(
        "aleator:sortire",
        Valor::Lista(vec![Valor::Numerus(2), Valor::Numerus(4)]),
    ));
    let Valor::Numerus(n) = sorted.data else {
        panic!("sortire must return numerus");
    };
    assert!((2..=4).contains(&n));

    let bytes = kernel.route(&Frame::request_with("aleator:octetos", Valor::Numerus(4)));
    assert_eq!(bytes.status, Status::Byte);
    assert!(matches!(&bytes.data, Valor::Octeti(items) if items.len() == 4));

    let uuid = kernel.route(&Frame::request("aleator:uuid"));
    let Valor::Textus(uuid) = uuid.data else {
        panic!("uuid must return textus");
    };
    assert_eq!(uuid.len(), 36);
    assert_eq!(&uuid[14..15], "4");
}

#[test]
fn routes_tempus_clock_values() {
    let kernel = HostKernel::new();

    let now = kernel.route(&Frame::request("tempus:nunc"));
    assert_eq!(now.status, Status::Done);
    let instant = Instans::try_from_valor(&now.data, InstansPraecisio::Nanosecunda)
        .expect("tempus:nunc must return a nanosecond instans");
    assert_eq!(instant.praecisio(), InstansPraecisio::Nanosecunda);

    for route in ["tempus:monotonicum", "tempus:activum"] {
        let response = kernel.route(&Frame::request(route));
        assert_eq!(response.status, Status::Done);
        assert!(matches!(response.data, Valor::Numerus(n) if n >= 0));
    }
}

#[test]
fn cli_manifest_prints_host_echo() {
    let output = Command::new(env!("CARGO_BIN_EXE_faber-host-macos-arm64"))
        .arg("manifest")
        .output()
        .expect("failed to run host manifest command");

    assert!(output.status.success());
    let json: Value = serde_json::from_slice(&output.stdout).expect("manifest should be JSON");
    assert_eq!(json["host"], Value::String("macos-arm64".into()));
    assert!(json["builtins"]
        .as_array()
        .expect("builtins should be an array")
        .iter()
        .any(|item| item["name"] == Value::String("host:echo".into())));
    assert!(json["builtins"]
        .as_array()
        .expect("builtins should be an array")
        .iter()
        .any(|item| item["name"] == Value::String("consolum:scribe".into())));
}

#[test]
fn manifest_names_exact_registered_norma_routes() {
    let manifest = HostKernel::new().manifest();
    let routes = manifest
        .builtins
        .iter()
        .map(|syscall| syscall.name.as_str())
        .collect::<BTreeSet<_>>();

    for route in [
        "aleator:fractum",
        "consolum:scribe",
        "processus:exsequi",
        "solum:lege",
        "tempus:nunc",
    ] {
        assert!(routes.contains(route), "manifest missing {route}");
    }
}

#[test]
fn cli_unresolved_call_prints_no_route_frame() {
    let output = Command::new(env!("CARGO_BIN_EXE_faber-host-macos-arm64"))
        .args(["call", "pg:query", "{}"])
        .output()
        .expect("failed to run host call command");

    assert_eq!(output.status.code(), Some(2));
    let json: Value = serde_json::from_slice(&output.stdout).expect("response should be JSON");
    assert_eq!(json["status"], Value::String("error".into()));
    assert_eq!(json["data"]["code"], Value::String("E_NO_ROUTE".into()));
}

fn round_trip(data: &Valor) -> Valor {
    let json = valor_to_json(data).expect("encode should succeed");
    faber_host_macos_arm64::kernel::valor_wire::json_to_valor(json).expect("decode should succeed")
}

#[test]
fn valor_wire_round_trips_instans_with_tag() {
    let data = Valor::Instans("2026-07-20T01:58:34Z".into());
    let json = data_json(&data);
    assert_eq!(
        json["$instans"],
        Value::String("2026-07-20T01:58:34Z".into())
    );
    assert_eq!(round_trip(&data), data);
}

#[test]
fn valor_wire_round_trips_octeti_with_tag() {
    let data = Valor::Octeti(vec![0, 1, 127, 255]);
    let json = data_json(&data);
    assert_eq!(
        json["$octeti"],
        Value::Array(vec![0u64.into(), 1u64.into(), 127u64.into(), 255u64.into()])
    );
    assert_eq!(round_trip(&data), data);
}

#[test]
fn valor_wire_keeps_textus_shape_unchanged() {
    let data = Valor::Textus("2026-07-20T01:58:34Z".into());
    assert_eq!(
        data_json(&data),
        Value::String("2026-07-20T01:58:34Z".into())
    );
    assert_eq!(round_trip(&data), data);
}

#[test]
fn valor_wire_escapes_single_dollar_key_tabula() {
    let data = frame_data::tabula([("$instans", Valor::Textus("not-an-instans".into()))]);
    let json = data_json(&data);
    assert_eq!(
        json["$tabula"]["$instans"],
        Value::String("not-an-instans".into())
    );
    assert_eq!(round_trip(&data), data);
}

#[test]
fn valor_wire_allows_dollar_keys_in_multi_key_tabula() {
    let data = frame_data::tabula([
        ("$instans", Valor::Textus("plain".into())),
        ("other", Valor::Numerus(1)),
    ]);
    assert_eq!(round_trip(&data), data);
}

#[test]
fn valor_wire_rejects_unknown_dollar_tag() {
    let json = serde_json::json!({"$bogus": 1});
    let error = faber_host_macos_arm64::kernel::valor_wire::json_to_valor(json)
        .expect_err("unknown $ tag should be rejected");
    assert!(error.to_string().contains("$bogus"));
}

#[test]
fn valor_wire_frame_round_trips_typed_variants_through_serde() {
    let request = Frame::request_with(
        "host:echo",
        frame_data::tabula([
            ("when", Valor::Instans("2026-07-20T01:58:34Z".into())),
            ("blob", Valor::Octeti(vec![9, 8, 7])),
            ("label", Valor::Textus("salve".into())),
        ]),
    );

    let raw = serde_json::to_string(&request).expect("frame should serialize");
    let decoded: Frame = serde_json::from_str(&raw).expect("frame should deserialize");

    assert_eq!(decoded, request);
}
