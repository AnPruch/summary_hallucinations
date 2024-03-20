import json
import nltk
import summedits_lib.utils_diff as udif


def clear_edits(dataset: list[dict]) -> list[dict]:
    cleared_data = []
    # start_green = "\033[1;32m"
    # start_red = "\033[1;31m"
    start = "\033[1;3"
    end = "\033[0m"
    for data in dataset:
        marked = udif.make_colored_text(data["original_summary"], data["summary"])
        changes = []
        while '\033[' in marked:
            start_index = marked.index(start)
            end_index = marked.index(end)
            changes.append(marked[start_index + len(start) + 2:end_index])
            marked = marked[end_index + len(end):]

        cleared_data.append({'original_summary': data['original_summary'],
                             'changed_summary': data['summary'],
                             'changes': changes,
                             'edit_types': data['edit_types']})
    return cleared_data


def get_summedits(domain: str) -> tuple[list, list]:
    with open(f"summedits_lib/data/summedits/summedits_{domain}.json", "r") as f:
        domain_data = json.load(f)
    changed_data = [data for data in domain_data
                    if data['label'] == 0]

    consistent_edits = [data for data in domain_data
                        if data['label'] == 1 and data['edit_types']]

    cleared_data = clear_edits(changed_data)
    consistent_clear = clear_edits(consistent_edits)

    return cleared_data, consistent_clear


def get_all_domain_edit_types(domain_data: list[tuple]) -> dict:
    edit_types = []
    for pair in domain_data:
        for edit in pair[0]:
            edit_types.append(edit)

    domain_edits = {}
    for edit in set(edit_types):
        domain_edits[edit] = edit_types.count(edit)
    return domain_edits


def compare_domain_edit_types(domain_types: dict[str, dict]) -> None:
    for domain, types in domain_types.items():
        print(f'{domain}: {types}')


def get_edits(data: list[tuple]) -> dict:
    all_changes = {}

    for (types, changes) in sorted(data):
        edit_types = tuple(types)
        if edit_types not in all_changes:
            all_changes[edit_types] = changes
        else:
            all_changes[edit_types].extend([';'] + changes)

    return all_changes


def tag_pos(text: str) -> dict:
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    parts_of_speech = {}
    for word, tag in tagged:
        if not tag.isalpha():
            continue
        if tag not in parts_of_speech:
            parts_of_speech[tag] = 0
        parts_of_speech[tag] += 1
    return parts_of_speech


def relative_freq_pos(parts_of_speech: dict) -> dict:
    all_token_count = sum(parts_of_speech.values())
    r_freq_pos = {}
    for pos, freq in parts_of_speech.items():
        r_freq_pos[pos] = freq/all_token_count
    return r_freq_pos


def sort_pos_freq(pos: dict) -> dict:
    pos_list = [(parts, freq) for parts, freq in pos.items()]
    sorted_pos = sorted(pos_list, key=lambda x: x[1], reverse=True)
    return dict(sorted_pos)


if __name__ == '__main__':
    domains = ["podcast", "billsum", "samsum", "news", "sales_call", "sales_email",
                "shakespeare", "scitldr", "qmsumm", "ectsum"]

    all_edits = {}
    consistent_edits = {}
    edit_types = {}
    for domain in domains:
        edits, consistent = get_summedits(domain)
        all_edits[domain] = []
        consistent_edits[domain] = []
        for data in edits:
            all_edits[domain].append((data['edit_types'], data['changes']))
        for data in consistent:
            consistent_edits[domain].append((data['edit_types'], data['changes']))

        edit_types[domain] = get_all_domain_edit_types(all_edits[domain])

    print('Compare which alterations are used for each dataset: ')
    compare_domain_edit_types(edit_types)
    print('\n0 - inconsistent, 1 - consistent')

    for domain in domains:
        edits = get_edits(all_edits[domain])
        with open(f'SummEdits Files/{domain}.txt', 'w', encoding='utf-8') as file:
            for types, edits in edits.items():
                changes = ', '.join(edits)
                file.write(', '.join(types) + '\n')
                file.write(changes + '\n')
                tags = tag_pos(changes)
                tags = sort_pos_freq(tags)
                relative_tags = relative_freq_pos(tags)
                relative_tags = sort_pos_freq(relative_tags)
                file.write(str(tags) + '\n' + str(relative_tags) + '\n\n')

    for domain, edits in consistent_edits.items():
        print(domain, edits)

