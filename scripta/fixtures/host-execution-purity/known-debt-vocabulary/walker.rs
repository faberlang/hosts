fn check_counts(n_kv: u64) -> HostResult<()> {
    if n_kv > 4096 {
        return Err(HostError::invalid_args(format!(
            "walker GGUF metadata count {n_kv} exceeds the bounded ceiling"
        )));
    }
    Ok(())
}

fn header_magic() -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes
}

fn admit(bytes: &[u8], map: &BTreeMap<u32, WeightFileRange>) -> HostResult<Inputs> {
    let inputs = inputs_from_gguf(bytes, map)?;
    Ok(inputs)
}
