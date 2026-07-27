use faber::Valor;
use host_kernel::{parse_manifest, Kernel};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

struct ProviderCase {
    name: &'static str,
    prefix: &'static str,
    manifest_json: &'static str,
    register: fn(&mut Kernel) -> host_kernel::HostResult<()>,
    provider: fn() -> host_kernel::HostResult<Box<dyn host_kernel::Provider>>,
    public_routes: &'static [&'static str],
    excluded_routes: &'static [&'static str],
}

struct DispatchFixture {
    opener: Valor,
    target: Option<String>,
    cancelled: bool,
}

impl DispatchFixture {
    fn new(opener: Valor) -> Self {
        Self {
            opener,
            target: None,
            cancelled: false,
        }
    }

    fn cancelled(opener: Valor) -> Self {
        Self {
            opener,
            target: None,
            cancelled: true,
        }
    }
}

struct TestWorkspace {
    root: PathBuf,
    next_id: usize,
}

impl TestWorkspace {
    fn new(provider: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "faber-host-provider-contracts-{provider}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("create test workspace");
        Self { root, next_id: 0 }
    }

    fn file(&mut self, name: &str, contents: impl AsRef<[u8]>) -> String {
        let path = self.unique_path(name);
        std::fs::write(&path, contents).expect("write fixture file");
        path.to_string_lossy().into_owned()
    }

    fn dir(&mut self, name: &str) -> String {
        let path = self.unique_path(name);
        std::fs::create_dir_all(&path).expect("create fixture directory");
        path.to_string_lossy().into_owned()
    }

    fn path(&mut self, name: &str) -> String {
        self.unique_path(name).to_string_lossy().into_owned()
    }

    fn unique_path(&mut self, name: &str) -> PathBuf {
        self.next_id += 1;
        self.root.join(format!("{}-{name}", self.next_id))
    }
}

impl Drop for TestWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

const ALEATOR_ROUTES: &[&str] = &[
    "aleator:fractum",
    "aleator:sortire",
    "aleator:octetos",
    "aleator:uuid",
    "aleator:semina",
];

const CONSOLUM_ROUTES: &[&str] = &[
    "consolum:hauri",
    "consolum:hauriet",
    "consolum:lege",
    "consolum:leget",
    "consolum:funde",
    "consolum:scribe",
    "consolum:scribet",
    "consolum:dic",
    "consolum:dicet",
    "consolum:mone",
    "consolum:monet",
    "consolum:vide",
    "consolum:videbit",
    "consolum:audit",
    "consolum:loquitur",
    "consolum:admonet",
];

const PROCESSUS_ROUTES: &[&str] = &[
    "processus:exsequi",
    "processus:exsequetur",
    "processus:dimitte",
    "processus:lege",
    "processus:scribe",
    "processus:sedes",
    "processus:muta",
    "processus:identitas",
    "processus:argumenta",
    "processus:captura",
];

const SOLUM_ROUTES: &[&str] = &[
    "solum:lege",
    "solum:hauri",
    "solum:hauriet",
    "solum:partem",
    "solum:inveni",
    "solum:carpe",
    "solum:carpiet",
    "solum:scribe",
    "solum:scribet",
    "solum:funde",
    "solum:appone",
    "solum:apponet",
    "solum:exstat",
    "solum:exstabit",
    "solum:directoriumne",
    "solum:regularene",
    "solum:legibilene",
    "solum:vinculumne",
    "solum:mensura",
    "solum:modum",
    "solum:modus",
    "solum:vincula",
    "solum:dele",
    "solum:delet",
    "solum:exscribe",
    "solum:exscribet",
    "solum:renomina",
    "solum:renominabit",
    "solum:tange",
    "solum:tanget",
    "solum:sequere",
    "solum:sequetur",
    "solum:crea",
    "solum:creabit",
    "solum:enumera",
    "solum:enumerabit",
    "solum:amputa",
    "solum:amputabit",
    "solum:domus",
    "solum:temporarium",
    "solum:iunge",
    "solum:parens",
    "solum:nomen",
    "solum:suffixum",
    "solum:absolve",
];

const TEMPUS_ROUTES: &[&str] = &[
    "tempus:nunc",
    "tempus:monotonicum",
    "tempus:activum",
    "tempus:dormiet",
];

fn aleator_provider() -> host_kernel::HostResult<Box<dyn host_kernel::Provider>> {
    Ok(Box::new(aleator::Aleator::new()?))
}

fn consolum_provider() -> host_kernel::HostResult<Box<dyn host_kernel::Provider>> {
    Ok(Box::new(consolum::Consolum::with_line_reader_for_tests(
        Cursor::new(b"contract-standalone\n".to_vec()),
    )?))
}

fn register_consolum_contract_provider(kernel: &mut Kernel) -> host_kernel::HostResult<()> {
    kernel.register(Arc::new(consolum::Consolum::with_line_reader_for_tests(
        Cursor::new(b"contract-lege\ncontract-leget\n".to_vec()),
    )?))
}

fn processus_provider() -> host_kernel::HostResult<Box<dyn host_kernel::Provider>> {
    Ok(Box::new(processus::Processus::new()?))
}

fn solum_provider() -> host_kernel::HostResult<Box<dyn host_kernel::Provider>> {
    Ok(Box::new(solum::Solum::new()?))
}

fn tempus_provider() -> host_kernel::HostResult<Box<dyn host_kernel::Provider>> {
    Ok(Box::new(tempus::Tempus::new()?))
}

fn provider_cases() -> [ProviderCase; 5] {
    [
        ProviderCase {
            name: "aleator",
            prefix: "aleator",
            manifest_json: aleator::manifest_json(),
            register: aleator::register,
            provider: aleator_provider,
            public_routes: ALEATOR_ROUTES,
            excluded_routes: &[],
        },
        ProviderCase {
            name: "consolum",
            prefix: "consolum",
            manifest_json: consolum::manifest_json(),
            register: register_consolum_contract_provider,
            provider: consolum_provider,
            public_routes: CONSOLUM_ROUTES,
            excluded_routes: &["consolum:fundet"],
        },
        ProviderCase {
            name: "processus",
            prefix: "processus",
            manifest_json: processus::manifest_json(),
            register: processus::register,
            provider: processus_provider,
            public_routes: PROCESSUS_ROUTES,
            excluded_routes: &["processus:exi"],
        },
        ProviderCase {
            name: "solum",
            prefix: "solum",
            manifest_json: solum::manifest_json(),
            register: solum::register,
            provider: solum_provider,
            public_routes: SOLUM_ROUTES,
            excluded_routes: &["solum:fundet", "solum:leget"],
        },
        ProviderCase {
            name: "tempus",
            prefix: "tempus",
            manifest_json: tempus::manifest_json(),
            register: tempus::register,
            provider: tempus_provider,
            public_routes: TEMPUS_ROUTES,
            excluded_routes: &["tempus:expectet"],
        },
    ]
}

fn dispatch_context(cancelled: bool) -> host_kernel::DispatchContext {
    host_kernel::DispatchContext {
        cancellation: host_kernel::CancellationProbe::new(move || cancelled),
    }
}

fn request(
    route: &str,
    fixture: DispatchFixture,
) -> (host_kernel::RequestFrame, host_kernel::DispatchContext) {
    (
        host_kernel::RequestFrame {
            conversation_id: format!("contract-{route}"),
            route: route.to_owned(),
            opener: fixture.opener,
            target: fixture.target,
        },
        dispatch_context(fixture.cancelled),
    )
}

// Fixture table intentionally repeats Valor openers across routes and is long
// by route-coverage design; keep map readable over clippy style limits.
#[allow(clippy::too_many_lines, clippy::match_same_arms)]
fn public_fixture(route: &str, workspace: &mut TestWorkspace) -> DispatchFixture {
    match route {
        "aleator:fractum" | "aleator:uuid" => DispatchFixture::new(Valor::Nihil),
        "aleator:sortire" => {
            DispatchFixture::new(Valor::Lista(vec![Valor::Numerus(0), Valor::Numerus(0)]))
        }
        "aleator:octetos" => DispatchFixture::new(Valor::Numerus(0)),
        "aleator:semina" => DispatchFixture::new(Valor::Numerus(1)),

        "consolum:hauri" | "consolum:hauriet" => DispatchFixture::new(Valor::Numerus(0)),
        "consolum:lege" | "consolum:leget" => DispatchFixture::new(Valor::Nihil),
        "consolum:funde" => DispatchFixture::new(Valor::Octeti(Vec::new())),
        "consolum:scribe" | "consolum:scribet" | "consolum:dic" | "consolum:dicet"
        | "consolum:mone" | "consolum:monet" | "consolum:vide" | "consolum:videbit" => {
            DispatchFixture::new(Valor::Textus(String::new()))
        }
        "consolum:audit" | "consolum:loquitur" | "consolum:admonet" => {
            DispatchFixture::new(Valor::Nihil)
        }

        "processus:exsequi" | "processus:exsequetur" => {
            DispatchFixture::new(Valor::Textus("printf ok".to_owned()))
        }
        "processus:dimitte" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus("sh".to_owned()),
            Valor::Textus("-c".to_owned()),
            Valor::Textus("true".to_owned()),
        ])),
        "processus:lege" => DispatchFixture::new(Valor::Textus("PATH".to_owned())),
        "processus:scribe" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus(format!("FABER_PROVIDER_CONTRACTS_{}", std::process::id())),
            Valor::Textus("ok".to_owned()),
        ])),
        "processus:sedes" | "processus:identitas" | "processus:argumenta" => {
            DispatchFixture::new(Valor::Nihil)
        }
        "processus:muta" => DispatchFixture::new(Valor::Textus(
            std::env::current_dir()
                .expect("current dir")
                .to_string_lossy()
                .into_owned(),
        )),
        "processus:captura" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus("sh".to_owned()),
            Valor::Textus("-c".to_owned()),
            Valor::Textus("printf ok".to_owned()),
        ])),

        "solum:lege"
        | "solum:hauri"
        | "solum:hauriet"
        | "solum:carpe"
        | "solum:carpiet"
        | "solum:exstat"
        | "solum:exstabit"
        | "solum:directoriumne"
        | "solum:regularene"
        | "solum:legibilene"
        | "solum:vinculumne"
        | "solum:mensura"
        | "solum:modus"
        | "solum:absolve" => {
            DispatchFixture::new(Valor::Textus(workspace.file("file.txt", "alpha\nbeta\n")))
        }
        "solum:partem" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus(workspace.file("range.txt", "alpha")),
            Valor::Numerus(0),
            Valor::Numerus(2),
        ])),
        "solum:inveni" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus(workspace.file("find.txt", "alpha")),
            Valor::Textus("ph".to_owned()),
            Valor::Numerus(0),
            Valor::Numerus(5),
        ])),
        "solum:scribe" | "solum:scribet" | "solum:appone" | "solum:apponet" => {
            DispatchFixture::new(Valor::Lista(vec![
                Valor::Textus(workspace.path("write.txt")),
                Valor::Textus("ok".to_owned()),
            ]))
        }
        "solum:funde" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus(workspace.path("bytes.bin")),
            Valor::Octeti(vec![1, 2, 3]),
        ])),
        "solum:modum" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus(workspace.file("mode.txt", "mode")),
            Valor::Numerus(0o600),
        ])),
        "solum:vincula" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus(workspace.file("symlink-source.txt", "link")),
            Valor::Textus(workspace.path("symlink-link.txt")),
        ])),
        "solum:dele" | "solum:delet" => {
            DispatchFixture::new(Valor::Textus(workspace.file("delete.txt", "delete")))
        }
        "solum:exscribe" | "solum:exscribet" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus(workspace.file("copy-source.txt", "copy")),
            Valor::Textus(workspace.path("copy-dest.txt")),
        ])),
        "solum:renomina" | "solum:renominabit" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus(workspace.file("rename-source.txt", "rename")),
            Valor::Textus(workspace.path("rename-dest.txt")),
        ])),
        "solum:tange" | "solum:tanget" => {
            DispatchFixture::new(Valor::Textus(workspace.path("touch.txt")))
        }
        "solum:sequere" | "solum:sequetur" => {
            let source = workspace.file("follow-source.txt", "follow");
            let link = workspace.path("follow-link.txt");
            std::os::unix::fs::symlink(&source, &link).expect("symlink fixture");
            DispatchFixture::new(Valor::Textus(link))
        }
        "solum:crea" | "solum:creabit" | "solum:amputa" | "solum:amputabit" => {
            DispatchFixture::new(Valor::Textus(workspace.dir("dir")))
        }
        "solum:enumera" | "solum:enumerabit" => {
            DispatchFixture::new(Valor::Textus(workspace.dir("list-dir")))
        }
        "solum:domus" | "solum:temporarium" => DispatchFixture::new(Valor::Nihil),
        "solum:iunge" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus("a".to_owned()),
            Valor::Textus("b".to_owned()),
        ])),
        "solum:parens" | "solum:nomen" | "solum:suffixum" => {
            DispatchFixture::new(Valor::Textus("a/b.txt".to_owned()))
        }

        "tempus:nunc" | "tempus:monotonicum" | "tempus:activum" => {
            DispatchFixture::new(Valor::Nihil)
        }
        "tempus:dormiet" => DispatchFixture::new(Valor::Numerus(0)),
        other => panic!("missing public dispatch fixture for {other}"),
    }
}

fn excluded_fixture(route: &str) -> DispatchFixture {
    match route {
        "consolum:fundet" => DispatchFixture::new(Valor::Octeti(Vec::new())),
        "processus:exi" | "tempus:expectet" => DispatchFixture::new(Valor::Numerus(0)),
        "solum:fundet" => DispatchFixture::new(Valor::Lista(vec![
            Valor::Textus("ignored".to_owned()),
            Valor::Octeti(Vec::new()),
        ])),
        "solum:leget" => DispatchFixture::cancelled(Valor::Textus("ignored".to_owned())),
        other => panic!("missing excluded dispatch fixture for {other}"),
    }
}

#[test]
fn solum_lege_returns_textus_content() {
    let mut kernel = Kernel::new();
    solum::register(&mut kernel).expect("register solum");
    let mut workspace = TestWorkspace::new("solum-lege-textus");
    let path = workspace.file("read.txt", "prima\nsecunda\n");

    let (request, context) = request(
        "solum:lege",
        DispatchFixture {
            opener: Valor::Textus(path),
            target: Some(std::any::type_name::<String>().to_owned()),
            cancelled: false,
        },
    );
    let reply = kernel
        .dispatch(&request, &context)
        .expect("solum:lege textus dispatch");
    assert!(
        matches!(
            reply.contents.as_slice(),
            [host_kernel::ProviderContent::Item(Valor::Textus(text))] if text == "prima\nsecunda\n"
        ),
        "solum:lege must satisfy its textus manifest result, got {reply:?}"
    );
}

#[test]
fn solum_lege_rejects_non_text_targets() {
    let mut kernel = Kernel::new();
    solum::register(&mut kernel).expect("register solum");
    let mut workspace = TestWorkspace::new("solum-lege-reject");
    let path = workspace.file("read.txt", "prima\nsecunda\n");

    for target in [
        std::any::type_name::<Vec<String>>(),
        std::any::type_name::<Vec<u8>>(),
    ] {
        let (request, context) = request(
            "solum:lege",
            DispatchFixture {
                opener: Valor::Textus(path.clone()),
                target: Some(target.to_owned()),
                cancelled: false,
            },
        );
        let error = kernel
            .dispatch(&request, &context)
            .expect_err("solum:lege must reject non-text targets before returning frames");
        assert_eq!(error.code, "E_INTERNAL");
        assert!(
            error.message.contains("use solum:carpe") && error.message.contains("solum:hauri"),
            "solum:lege target error must point at manifest routes, got {error:?}"
        );
    }
}

#[test]
fn solum_carpe_returns_list_of_lines() {
    let mut kernel = Kernel::new();
    solum::register(&mut kernel).expect("register solum");
    let mut workspace = TestWorkspace::new("solum-carpe-list");
    let path = workspace.file("read.txt", "prima\nsecunda\n");

    let (request, context) =
        request("solum:carpe", DispatchFixture::new(Valor::Textus(path)));
    let reply = kernel
        .dispatch(&request, &context)
        .expect("solum:carpe lista<textus> dispatch");
    assert_eq!(
        reply.contents.as_slice(),
        &[
            host_kernel::ProviderContent::Item(Valor::Textus("prima".to_owned())),
            host_kernel::ProviderContent::Item(Valor::Textus("secunda".to_owned())),
        ],
        "solum:carpe must carry the list contract formerly claimed by solum:lege"
    );
}

#[test]
fn solum_hauri_returns_raw_bytes() {
    let mut kernel = Kernel::new();
    solum::register(&mut kernel).expect("register solum");
    let mut workspace = TestWorkspace::new("solum-hauri-bytes");
    let path = workspace.file("read.txt", "prima\nsecunda\n");

    let (request, context) =
        request("solum:hauri", DispatchFixture::new(Valor::Textus(path)));
    let reply = kernel
        .dispatch(&request, &context)
        .expect("solum:hauri octeti dispatch");
    assert!(
        matches!(reply.contents.as_slice(), [host_kernel::ProviderContent::Byte(bytes)] if bytes == b"prima\nsecunda\n"),
        "solum:hauri must carry the byte contract formerly claimed by solum:lege, got {reply:?}"
    );
}

#[test]
fn composed_kernel_has_correct_provider_identities() {
    let cases = provider_cases();
    let mut kernel = Kernel::new();
    for case in &cases {
        (case.register)(&mut kernel)
            .unwrap_or_else(|error| panic!("register {}: {error}", case.name));
    }

    let manifest = kernel.manifest();
    assert_eq!(manifest.providers.len(), cases.len());

    let expected_names = cases.iter().map(|case| case.name).collect::<BTreeSet<_>>();
    let expected_prefixes = cases
        .iter()
        .map(|case| case.prefix)
        .collect::<BTreeSet<_>>();
    let actual_names = manifest
        .providers
        .iter()
        .map(|provider| provider.provider.as_str())
        .collect::<BTreeSet<_>>();
    let actual_prefixes = manifest
        .providers
        .iter()
        .flat_map(|provider| provider.prefixes.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();

    assert_eq!(actual_names, expected_names);
    assert_eq!(actual_prefixes, expected_prefixes);
}

#[test]
#[allow(clippy::too_many_lines)]
fn composed_kernel_registers_unique_provider_identities_and_routes() {
    let cases = provider_cases();
    let mut kernel = Kernel::new();
    for case in &cases {
        (case.register)(&mut kernel)
            .unwrap_or_else(|error| panic!("register {}: {error}", case.name));
    }

    let manifest = kernel.manifest();
    let providers_by_name = manifest
        .providers
        .iter()
        .map(|provider| (provider.provider.as_str(), provider))
        .collect::<BTreeMap<_, _>>();
    let mut admitted_routes = BTreeSet::new();

    for case in &cases {
        let provider = (case.provider)()
            .unwrap_or_else(|error| panic!("create {} provider: {error}", case.name));
        let mut workspace = TestWorkspace::new(case.name);
        let standalone = parse_manifest(case.manifest_json)
            .unwrap_or_else(|error| panic!("parse {} manifest: {error}", case.name));
        let composed = providers_by_name
            .get(case.name)
            .unwrap_or_else(|| panic!("composed manifest missing {}", case.name));

        assert_eq!(composed.prefixes, vec![case.prefix.to_owned()]);
        assert_eq!(composed.calls, standalone.calls);

        let manifest_routes = standalone
            .calls
            .iter()
            .map(|call| call.route.as_str())
            .collect::<BTreeSet<_>>();
        let expected_routes = case.public_routes.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(
            manifest_routes, expected_routes,
            "{} route bijection",
            case.name
        );

        for route in case.public_routes {
            assert!(admitted_routes.insert(*route), "duplicate route {route}");
            assert!(kernel.supports_route(route), "kernel should admit {route}");
            if matches!(*route, "consolum:lege" | "consolum:leget") {
                let call = standalone
                    .calls
                    .iter()
                    .find(|call| call.route == *route)
                    .unwrap_or_else(|| panic!("manifest missing {route}"));
                assert_eq!(call.result, "textus");
            }
            let fixture = public_fixture(route, &mut workspace);
            let expect_cancelled = fixture.cancelled;
            let (request, context) = request(route, fixture);
            if expect_cancelled {
                let result = provider.dispatch(&request, &context);
                assert!(
                    !matches!(&result, Err(error) if error.code == "E_NO_ROUTE"),
                    "{} manifest route {route} must reach Provider::dispatch, got {result:?}",
                    case.name
                );
            } else {
                let result = kernel.dispatch(&request, &context);
                assert!(
                    result.is_ok(),
                    "{} manifest route {route} must satisfy kernel dispatch with safe fixture, got {result:?}",
                    case.name
                );
                if matches!(*route, "consolum:lege" | "consolum:leget") {
                    let reply = result.expect("line read result");
                    assert!(
                        matches!(
                            reply.contents.as_slice(),
                            [host_kernel::ProviderContent::Item(Valor::Textus(_))]
                        ),
                        "{route} must return one text item, got {reply:?}"
                    );
                }
            }
        }
        for route in case.excluded_routes {
            assert!(
                !manifest_routes.contains(route),
                "{} should not manifest {route}",
                case.name
            );
            assert!(!kernel.supports_route(route), "kernel should deny {route}");
            let (request, context) = request(route, excluded_fixture(route));
            let error = match provider.dispatch(&request, &context) {
                Ok(reply) => panic!(
                    "{} should reject excluded route {route}, got {reply:?}",
                    case.name
                ),
                Err(error) => error,
            };
            assert_eq!(
                error.code, "E_NO_ROUTE",
                "{} excluded route {route} must be rejected by dispatch",
                case.name
            );
        }
    }

    assert_eq!(admitted_routes.len(), 80);
    std::env::remove_var(format!("FABER_PROVIDER_CONTRACTS_{}", std::process::id()));
}
