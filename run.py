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


FIELDS = ['date','username','id','quality','doc_type', 'publish_year','lang_fasttext','lang_fasttext_conf','text']


def add_data(example, quality, username):
    example['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    example['username'] = username
    example['quality'] = quality
    output = [example.get(key) for key in FIELDS]
    print(output)
    sh.append_row(output)


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
if 'examples' not in st.session_state:
    st.session_state["examples"] = []
    for i in range(100):
        st.session_state["examples"].append(next(st.session_state["iter"]))

st.title(dataset)
st.markdown("For labelling the quality of NCC. Fill in a name and get started!")
username = st.text_input("Your name", help="We use this to track who has labeled what")

st.markdown("""---""")

if username:
    if len(st.session_state["examples"]) > 0:
        example = st.session_state["examples"][0]
        st.write(example)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Crap"):
                st.session_state["examples"].pop(0)
                add_data(example, 0, username)
                st.rerun()
        with col2:
            if st.button("Medicore"):
                st.session_state["examples"].pop(0)
                add_data(example, 1, username)
                st.rerun()
        with col3:
            if st.button("Good"):
                st.session_state["examples"].pop(0)
                add_data(example, 2, username)
                st.rerun()
        with col4:
            if st.button("Skip"):
                st.session_state["examples"].pop(0)
                st.rerun()
    else:
        st.write("Congrats you just labaled 100 examples! Refresh to get more")


else:
    st.markdown("""Here we need to add some text about how to label correctly\n * saasdad\n * adsasd""")