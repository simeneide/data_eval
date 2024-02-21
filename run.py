#%%
from datasets import load_dataset
import streamlit as st
import json
import re
from datetime import datetime
import tiktoken

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
skip_doc_types = ['newspapers_online_nb', 'newspapers_online_nn']

FIELDS = ['date','username','id','quality','doc_type', 'text']


def add_data(example):
    example['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = [example.get(key) for key in FIELDS]
    print(example)
    sh.append_row(output)


def limit_tokens(string: str, max_tokens: int = 512, encoding_name: str = "gpt2") -> str:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string)) 
    if num_tokens <= max_tokens:
        return string
    sentence_array = string.split('.')
    while len(encoding.encode('.'.join(sentence_array))) > max_tokens:
        sentence_array.pop()

    return '.'.join(sentence_array)

#### functions from NB scriot

# compile regexes
username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
url_regex = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))')
email_regex = re.compile(r'[\w\.-]+@[\w\.-]+')


def replace_usernames_tweets(text, filler='@User'):
    # replace other user handles by filler
    text = str(text)
    text = re.sub(username_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split())
    return text


def replace_urls(text, filler='http://www.no'):
    text = str(text)
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
    text = re.sub(email_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split())
    return text

### end functions from NB script


def filter_function(example):
    return example['doc_type'] not in skip_doc_types


def mapping_function(example):
    example['text'] = replace_usernames_tweets(example['text'])
    example['text'] = replace_urls(example['text'])
    example['text'] = replace_email_addresses(example['text'])
    return example


if 'iter' not in st.session_state:
    data = load_dataset(
        dataset,
        split="train",
        streaming=True,
        use_auth_token=st.secrets["hf_token"]
        )
    data = data.shuffle(buffer_size=20000)
    data = data.filter(filter_function)
    data = data.map(mapping_function)
    st.session_state["data"] = data
    iter = iter(data)
    st.session_state["iter"] = iter

if 'example' not in st.session_state or not st.session_state["example"]:
    example = next(st.session_state["iter"])
    example['text'] = limit_tokens(example['text'])
    st.session_state["example"] = example

st.title(dataset)
st.markdown(f"For labelling the quality of {dataset}. Fill in a name and get started!")
username = st.text_input("Your name", help="We use this to track who has labeled what")

st.markdown("""---""")

if username:
    example = st.session_state["example"]
    st.text_area(f"Text Example:", value=example['text'], height=300, max_chars=None, key=None)

    col1, col2, col3 = st.columns(3)

    def handle_label(quality):
        example['username'] = username
        example['quality'] = quality
        add_data(example)  # Save the current example
        st.session_state["example"] = None
        st.rerun()

    with col1:
        if st.button("Crap"):
            handle_label(0)
    with col2:
        if st.button("Medicore"):
            handle_label(1)
    with col3:
        if st.button("Good"):
            handle_label(2)

    st.markdown(
"""The LANGUAGE or LENGTH of text does not matter. It is the quality of the text that matters:
* GOOD: The text is natural, coherent and readable. Like in a page of a book, blog or news article. No encoding errors or wierd characters.
* MEDICORE: The text is readable, but does not have a natural flow. It does have some coherent sentences. Like you would expect from a catalog, technical manual or other non-full text with parsed tables etc. 
* CRAP: The text is not coherent. It is either gibberish or has encoding errors. 
""")
        


else:
    st.markdown(
"""Here is how to label the examples correctly. The LANGUAGE or LENGTH of text does not matter. It is the quality of the text that matters:
* GOOD: The text is natural, coherent and readable. Like in a page of a book, blog or news article. No encoding errors or wierd characters.
* MEDICORE: The text is readable, but does not have a natural flow. It does have some coherent sentences. Like you would expect from a catalog, technical manual or reserach paper. 
* CRAP: The text is not coherent. It is either gibberish or has encoding errors. 
""")
   
