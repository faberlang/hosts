use super::*;

#[test]
fn builds_response_carrier() {
    let response = Replicatio::new(
        201,
        b"{\"ok\":true}".to_vec(),
        HashMap::from([("X-Faber-Test".to_owned(), "yes".to_owned())]),
    );

    assert_eq!(response.status(), 201);
    assert_eq!(
        response.caput("x-faber-test".to_owned()),
        Some("yes".to_owned())
    );
    assert!(matches!(response.corpus_json(), Valor::Tabula(_)));
    assert!(response.bene());
}

#[test]
fn error_response_carrier_has_correct_fields() {
    let response = Replicatio::new(
        500,
        b"internal error".to_vec(),
        HashMap::from([("x-faber-error".to_owned(), "http-client".to_owned())]),
    );

    assert_eq!(response.status(), 500);
    assert_eq!(response.corpus(), "internal error");
    assert!(!response.bene());
    assert_eq!(
        response.caput("x-faber-error".to_owned()),
        Some("http-client".to_owned())
    );
}

#[test]
fn empty_body_response_produces_nihil_json() {
    let response = Replicatio::new(200, b"".to_vec(), HashMap::new());

    assert_eq!(response.status(), 200);
    assert_eq!(response.corpus(), "");
    assert!(response.bene());
    assert_eq!(response.corpus_json(), Valor::Nihil);
}

#[test]
fn error_status_not_between_200_and_299() {
    for status in [100, 199, 300, 404, 599] {
        let response = Replicatio::new(status, b"".to_vec(), HashMap::new());
        assert!(!response.bene(), "status {status} should not be bene");
    }
}

#[test]
fn header_lookup_is_case_insensitive() {
    let response = Replicatio::new(
        200,
        b"ok".to_vec(),
        HashMap::from([("X-Custom-Header".to_owned(), "value".to_owned())]),
    );

    assert_eq!(
        response.caput("x-custom-header".to_owned()),
        Some("value".to_owned())
    );
    assert_eq!(
        response.caput("X-CUSTOM-HEADER".to_owned()),
        Some("value".to_owned())
    );
    assert_eq!(
        response.caput("X-Custom-Header".to_owned()),
        Some("value".to_owned())
    );
}

#[test]
fn missing_header_returns_none() {
    let response = Replicatio::new(200, b"ok".to_vec(), HashMap::new());
    assert_eq!(response.caput("nonexistent".to_owned()), None);
}

#[test]
fn status_code_is_exposed() {
    for status in [200, 201, 404, 500, 599] {
        let response = Replicatio::new(status, b"".to_vec(), HashMap::new());
        assert_eq!(response.status(), status);
    }
}

#[test]
fn text_corpus_preserves_body_as_string() {
    let response = Replicatio::new(200, b"hello world".to_vec(), HashMap::new());
    assert_eq!(response.corpus(), "hello world");
}

#[test]
fn binary_body_corpus_octeti_is_preserved() {
    let bytes: Vec<u8> = vec![0x00, 0x01, 0xFF, 0xFE];
    let response = Replicatio::new(200, bytes.clone(), HashMap::new());
    assert_eq!(response.corpus_octeti(), bytes);
}
