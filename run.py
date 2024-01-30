#%%
from datasets import load_dataset
import streamlit as st
import json
import re
from datetime import datetime

#%%
import gspread

gcloud_secret = {key: st.secrets[key] for key in ['type',
 'project_id',
 'private_key_id',
 'private_key',
 'client_email',
 'client_id',
 'auth_uri',
 'token_uri',
 'auth_provider_x509_cert_url',
 'client_x509_cert_url',
 'universe_domain']}

gc = gspread.service_account_from_dict(gcloud_secret)
sh = gc.open("ncc-data-eval").sheet1

#%%
dataset = "NbAiLab/NCC"

data = load_dataset(
    dataset,
    split="train",
    streaming=True,
    use_auth_token=st.secrets["hf_token"]
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

st.title(dataset)
st.markdown("For labelling the quality of NCC. Fill in a name and get started! (we should probably write some better guidelines here)")
username = st.text_input("Your name", help="We use this to track who has labeled what")

st.markdown("""---""")

if username != "":
    example = next(st.session_state["iter"])
    st.write(example)

col1, col2, col3 = st.columns(3)

FIELDS = ['date','username','id','quality','doc_type', 'publish_year','lang_fasttext','lang_fasttext_conf','text']
def add_data(example):
    example['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    example['username'] = username
    output = [example.get(key) for key in FIELDS]
    sh.append_row(output)

with col1:
    if st.button("Crap"):
        example["quality"] = "0"
        add_data(example)
with col2:
    if st.button("Medicore"):
        example["quality"] = "1"
        add_data(example)
with col3:
    if st.button("Good"):
        example["quality"] = "2"
        add_data(example)

#%%

# %%
