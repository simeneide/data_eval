from datasets import load_dataset
import streamlit as st
import json
import re

dataset = "NbAiLab/NCC"

data = load_dataset(
    dataset,
    split="train",
    streaming=True
    )


#### functions from NB scriot

# compile regexes
username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
url_regex = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))')
email_regex = re.compile(r'[\w\.-]+@[\w\.-]+')

def replace_usernames_tweets(text, filler='@User'):
    # replace other user handles by filler
    text = str(text)
    input_text = str(text)
    text = re.sub(username_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split())
    return text

def replace_urls(text, filler='http://www.no'):
    text = str(text)
    input_text = text
    # <url> is a marker used internally. use filler instead
    text = text.replace('<url>', filler)
    # replace other urls by filler
    text = re.sub(url_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split()) 
    return text

def replace_email_addresses(text, filler='email@email.no'):
    text = str(text)
    input_text = text
    text = re.sub(email_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split())
    return text

### end functions from NB script

def filter_function(example):
    return example['doc_type'] != 'newspaper_ocr'

def mapping_function(example):
    example['text'] = replace_usernames_tweets(example['text'])
    example['text'] = replace_urls(example['text'])
    example['text'] = replace_email_addresses(example['text'])
    return example

data = data.shuffle(buffer_size=10000)
data = data.filter(filter_function)
data = data.map(mapping_function)

if 'iter' not in st.session_state:
    iter = iter(data)
    st.session_state["iter"] = iter

if 'label_file' not in st.session_state:
    st.session_state["label_file"] = open("labels.jsonl", "w")

st.title(dataset)
example = next(st.session_state["iter"])
st.write(example)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Crap"):
        example["label"] = "0"
        st.session_state["label_file"].write(json.dumps(example) + "\n")
with col2:
    if st.button("Medicore"):
        example["label"] = "1"
        st.session_state["label_file"].write(json.dumps(example) + "\n")
with col3:
    if st.button("Good"):
        example["label"] = "2"
        st.session_state["label_file"].write(json.dumps(example) + "\n")


st.session_state["label_file"].flush()
