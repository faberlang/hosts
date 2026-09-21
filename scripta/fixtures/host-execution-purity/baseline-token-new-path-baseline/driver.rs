fn ssm_entry_guard(entry: &str) -> HostResult<()> {
    if entry != "SsmConv1d" {
        return Err(HostError::InvalidBind);
    }
    Ok(())
}
