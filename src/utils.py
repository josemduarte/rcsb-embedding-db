def alignment_url(query_id, target_id):
    query = query_id.split(".")[0]
    query_ch = query_id.split(".")[1]
    target = target_id.split(".")[0]
    target_ch = target_id.split(".")[1]
    return f"https://www.rcsb.org/alignment?request-body=%7B%22query%22%3A%7B%22options%22%3A%7B%22return_sequence_data%22%3Afalse%7D%2C%22context%22%3A%7B%22mode%22%3A%22pairwise%22%2C%22method%22%3A%7B%22name%22%3A%22tm-align%22%7D%2C%22structures%22%3A%5B%7B%22entry_id%22%3A%22{query}%22%2C%22selection%22%3A%7B%22asym_id%22%3A%22{query_ch}%22%7D%7D%2C%7B%22entry_id%22%3A%22{target}%22%2C%22selection%22%3A%7B%22asym_id%22%3A%22{target_ch}%22%7D%7D%5D%7D%7D%7D"


def img_url(instance_id):
    pdb = instance_id.split(".")[0].lower()
    ch = instance_id.split(".")[1]
    return f"https://cdn.rcsb.org/images/structures/{pdb}_chain-{ch}.jpeg"
