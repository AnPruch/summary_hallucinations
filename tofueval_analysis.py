from datasets import load_dataset
import json
import os
import pandas as pd


def get_dataset(topic_type: str) -> pd.DataFrame:
    directory = 'tofueval_lib/factual_consistency'
    datasets = [dataset for dataset in os.listdir(directory)
                if topic_type in dataset]
    df = pd.DataFrame()
    for dataset in datasets:
        with open(f"{directory}/{dataset}", 'r') as data:
            data = pd.read_csv(data)
            df = pd.concat([df, data])
    df = df.drop(columns=['doc_id', 'annotation_id', 'sent_idx'])

    category_dir = 'tofueval_lib/topic_category/'
    with open(f"{category_dir}{topic_type}_topic_category.json", 'r') as jfile:
        df_categories = json.load(jfile)

    df_categories = pd.json_normalize(df_categories).transpose()
    df_categories = df_categories.reset_index()
    df_categories = df_categories.rename(columns={'index': 'topic',
                                                  0: 'category'})
    df = pd.merge(df, df_categories, on='topic')

    return df


def calculate_percentages_for_types(halu: pd.DataFrame) -> pd.DataFrame:
    freq = {}
    for types in halu.type:
        splitted_types = types.split(', ')
        for tag in splitted_types:
            if tag not in freq:
                freq[tag] = 0
            freq[tag] += 1

    length = len(halu)
    percentages = ((tag, frequency / length) for tag, frequency in freq.items())
    return pd.DataFrame(percentages, columns=['type', '%'])


if __name__ == '__main__':
    mediadata = get_dataset('mediasum')
    meetingdata = get_dataset('meetingbank')

    halu_count_media = sum(mediadata.sent_label == 'no')
    halu_count_meet = sum(meetingdata.sent_label == 'no')
    for (halu_count, data) in ((halu_count_media, mediadata),
                                 (halu_count_meet, meetingdata)):
        all_data_count = len(data)
        print(f'{data=}\n'
              f'Count of hallucinating summaries in data: {halu_count}\n'
              f'Percent of hallucinating summaries in data: {halu_count /  all_data_count}\n\n')

    halu_media = mediadata[mediadata.sent_label == 'no']
    halu_media.to_csv('TofuEval Files/halu_media.csv')
    halu_meet = meetingdata[meetingdata.sent_label == 'no']
    halu_meet.to_csv('TofuEval Files/halu_meeting.csv')

    print('MediaSum', halu_media.type.value_counts(), sep='\n', end='\n\n')
    print('MeetingBank', halu_meet.type.value_counts(), sep='\n', end='\n\n')

    perc_media = calculate_percentages_for_types(halu_media)
    perc_meet = calculate_percentages_for_types(halu_meet)
    percentages = pd.merge(perc_media, perc_meet, on='type')
    percentages = percentages.rename(columns={'%_x': 'media',
                                              '%_y': 'meeting'})
    print(percentages)

    with open("tofueval_lib/document_ids_dev_test_split.json") as file:
        document_mapping = json.load(file)

    meetingbank_dev_ids = document_mapping['dev']['meetingbank']
    meetingbank_test_ids = document_mapping['test']['meetingbank']

    full_meetingbank = pd.DataFrame(load_dataset("lytang/MeetingBank-transcript")['test'])
    meetingbank_dev = full_meetingbank[full_meetingbank.meeting_id.isin(meetingbank_dev_ids)][
        ['meeting_id', 'source']].reset_index(drop=True).to_csv("TofuEval Files/meetingbank_dev_doc.csv", index=False)
    meetingbank_test = full_meetingbank[full_meetingbank.meeting_id.isin(meetingbank_test_ids)][
        ['meeting_id', 'source']].reset_index(drop=True).to_csv("TofuEval Files/meetingbank_test_doc.csv", index=False)
