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
n_examples = 100

data = load_dataset(
    dataset,
    split="train",
    streaming=True,
    use_auth_token=st.secrets["hf_token"]
    )


FIELDS = ['date','username','id','quality','doc_type']


def add_data(example, quality, username):
    example['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    example['username'] = username
    example['quality'] = quality
    output = [example.get(key) for key in FIELDS]
    print(example)
    sh.append_row(output)


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
    return example['doc_type'] != 'newspaper_ocr'

def mapping_function(example):
    example['text'] = replace_usernames_tweets(example['text'])
    example['text'] = replace_urls(example['text'])
    example['text'] = replace_email_addresses(example['text'])
    return example

data = data.shuffle(buffer_size=20000)
data = data.filter(filter_function)
data = data.map(mapping_function)

if 'iter' not in st.session_state:
    print('iter')
    iter = iter(data)
    st.session_state["iter"] = iter
if 'examples' not in st.session_state:
    print('examples')
    st.session_state["examples"] = []
    for i in range(n_examples):
        st.session_state["examples"].append(next(st.session_state["iter"]))
if 'n' not in st.session_state:
    st.session_state["n"] = 0

st.title(dataset)
st.markdown(f"For labelling the quality of {dataset}. Fill in a name and get started!")
username = st.text_input("Your name", help="We use this to track who has labeled what")

st.markdown("""---""")

if username:
    if st.session_state["n"] < n_examples:
        example = st.session_state["examples"][st.session_state["n"]]
        st.text_area(f"Text Example ({st.session_state['n']+1} of {n_examples}):", value=example['text'], height=300, max_chars=None, key=None)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Crap"):
                add_data(example, 0, username)
                st.session_state["n"] += 1
                st.rerun()
        with col2:
            if st.button("Medicore"):
                add_data(example, 1, username)
                st.session_state["n"] += 1
                st.rerun()
        with col3:
            if st.button("Good"):
                add_data(example, 2, username)
                st.session_state["n"] += 1
                st.rerun()
        st.markdown(
"""The LANGUAGE or LENGTH of text does not matter. It is the quality of the text that matters:
* GOOD: The text is natural, coherent and readable. Like in a page of a book, blog or news article. No encoding errors or wierd characters.
* MEDICORE: The text is readable, but does not have a natural flow. It does have some coherent sentences. Like you would expect from a catalog, technical manual or other non-full text with parsed tables etc. 
* CRAP: The text is not coherent. It is either gibberish or has encoding errors. 
""")
    else:
        st.write("Congrats you just labaled 100 examples! Refresh to get more")
        


else:
    st.markdown(
"""Here is how to label the examples correctly. The LANGUAGE or LENGTH of text does not matter. It is the quality of the text that matters:
* GOOD: The text is natural, coherent and readable. Like in a page of a book, blog or news article. No encoding errors or wierd characters.
* MEDICORE: The text is readable, but does not have a natural flow. It does have some coherent sentences. Like you would expect from a catalog, technical manual or reserach paper. 
* CRAP: The text is not coherent. It is either gibberish or has encoding errors. 
""")
