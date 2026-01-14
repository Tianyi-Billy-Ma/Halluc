def backtrack_generation(
    token_ids: list[int],
    backtrack_token_id: int,
    backtrack_count: int = 0,
):
    generated_token_ids, backtrack_count = [], 0
    for token_id in token_ids:
        if token_id == backtrack_token_id:
            backtrack_count += 1
        else:
            generated_token_ids = (
                generated_token_ids[:-backtrack_count]
                if backtrack_count > 0
                else generated_token_ids
            )
            generated_token_ids.append(token_id)
            backtrack_count = 0
    return generated_token_ids
