import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import networkx as nx
warnings.filterwarnings("ignore")
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_google_search_links(query):
    results = search(query)
    return [link for link in results]
def get_bing_search_links(query):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    ]
    url = 'https://www.bing.com/search?q={}'.format(query.replace(" ","+"))
    headers = {'User-Agent': user_agents[0]}
    r = requests.get(url,headers=headers)
    soup = BeautifulSoup(r.content, 'html5lib')
    results = soup.find("div",{'id':'b_content'})
    h2 = results.find_all("h2")
    li = []
    for i in h2:
        try:
            filter = i.find("a")["href"]
            if "https://" in filter:
                li.append(filter)
        except Exception as e:
            pass
    return li

def index_of_true(priority_tags):
    key_true = []
    for tag in priority_tags.keys():
        if priority_tags[tag] == True:
            key_true.append(tag)
        else:
            pass
    return key_true

def find_nested_links(tag):
    links = []
    nested_tags = tag.find_all('a')
    if len(nested_tags) != 0:
        for i in nested_tags:
            try:
                links.append(str(i).split(" ")[1].split("href=")[1]) #Storing the links
            except IndexError as index_error:
                print(index_error)
                return links
    return links

def adjust_tags_based_on_priority(priority_tags,tag_name,record_tags,index_value):
    #if less priority tags are already present and new high priority tag comes, we will make all low priority tag to false and high priority tag true
    #if less priority tags comes and high priority tag is True, We will keep all same and make low priority task true
    values = list(priority_tags.values())
    keys = list(priority_tags.keys())
    all_indexes_with_true = []
    for i in range(0,len(values)):
        if values[i]:
            all_indexes_with_true.append(i)
        else:
            pass
    if len(all_indexes_with_true) == 0:
        #If no tags are true, we will make the target tag true and return the same.
        priority_tags[tag_name] = not priority_tags[tag_name]
        record_tags[tag_name].append(index_value)
        return priority_tags,record_tags,""
    else:
        priority_type = "high" #high or low
        idx = keys.index(tag_name)
        if idx <= min(all_indexes_with_true):
            priority_type = "high"
            for k in keys:
                priority_tags[k] = False
            priority_tags[tag_name] = True
            record_tags[tag_name].append(index_value)
            return priority_tags,record_tags,"high"
        elif idx >= max(all_indexes_with_true):
            priority_type = "low"
            priority_tags[tag_name] = True
            record_tags[tag_name].append(index_value)
            return priority_tags,record_tags,"low"
        else:
            priority_type = "mid"
            # print('Index of mid tag',idx)
            for k in keys[idx+1:]:
                priority_tags[k] = False
            priority_tags[tag_name] = True
            record_tags[tag_name].append(index_value)
            return priority_tags,record_tags,"mid"
            #Make all the tags with less priority than current tag to false. And make the current tag true

def create_network_graph(indexing,sub_indexing,nxG,type,priority_tags,parent_node,record_tags):
    if type == "high":
        nxG.add_edge(parent_node,indexing)
    elif type == "low" or type == "mid":
        key_true = index_of_true(priority_tags)[:-1]
        # print("KGLA_structure --> index_of_true",key_true[-1])
        new_edge = record_tags[key_true[-1]][-1]
        nxG.add_edge(new_edge,indexing)
        for v in sub_indexing:
            nxG.add_edge(indexing, v)
        return nxG
    else:
        nxG.add_edge(parent_node,indexing)
    for v in sub_indexing:
            nxG.add_edge(indexing, v)
    return nxG
def clean_data(link):
    try:
        r = requests.get(link,timeout=(3, 5))
        print("request : {}".format(r))
        if r.status_code == 403:
            return [],""
        soup = BeautifulSoup(r.content, 'html5lib')
        h1_tag = str(soup.find('h1'))
        parent_node = soup.find('title').text       
        for tag in soup(['nav', 'header', 'footer', 'script', 'style', 'aside']):
            tag.decompose()
        imp_tags = soup.find_all(['h1', 'h2', 'h3', 'h4','h5', 'p', 'li','pre','img'])
        imp_tags.insert(0,BeautifulSoup(h1_tag, 'html5lib').find("h1"))
        return imp_tags,parent_node
    except Exception as e:
        print(e)
        return [],""

def generate_heading(hl,r,nxG):
    temp = []
    if len(hl) == 0:
        # print(r)
        return r
    else:
        for h in hl:
            # print(r)
            # print(list(nxG.adj[h]))
            temp.append(generate_heading(list(nxG.adj[h])[1:],r+'->'+str(h),nxG))
        return temp
    

def priority_based_structure(imp_tags,parent_node):
    try:
        priority_tags = {"h1":False,"h2":False,"h3":False,"h4":False,"h5":False,"p":False,"li":False,"pre":False,"a":False,"img":False}
        record_tags = {"h1":[],"h2":[],"h3":[],"h4":[],"h5":[],"li":[],"p":[],"a":[],"pre":[],"img":[]}
        text_index = {}
        indexing = 0
        links = []
        nxG = nx.Graph()
        nxG.add_node(parent_node)
        for idx,tag in enumerate(imp_tags):
            key_true = index_of_true(priority_tags)
            try:
                if tag == None:
                    continue
                if len(key_true) == 0 and tag.name in ["ul","li","ol"]: #If there is no heading, We will find the link and store them directly in a list for futher scraping
                    continue
                # links.extend(find_nested_links(tag))
                elif tag.name == "pre" or tag.name == "code":
                    text_index[indexing] = "<code>"+tag.text
                elif tag.name == "img":
                    if "https://" in tag["src"]:
                        text_index[indexing] = "<image>"+tag["src"]
                else:
                    text_index[indexing] = tag.text
            except Exception as e:
                print(e)
            sub_indexing = []
            priority_tags,record_tags,type = adjust_tags_based_on_priority(priority_tags,tag.name,record_tags,indexing)
            nxG = create_network_graph(indexing,sub_indexing,nxG,type,priority_tags,parent_node,record_tags)
            indexing = indexing+1
        return nxG,text_index
    except Exception as e:
        print(e)
        nxG = nx.Graph()
        text_index = {}
        return nxG,text_index


def start_network(links,store_data,topicName):
    extended_tags = []
    selected_links = 0
    for link in links:
        if selected_links > 5:
            break
        imp_tags,parent_node = clean_data(link)
        if len(imp_tags) != 0:
            extended_tags.extend(imp_tags)
            selected_links = selected_links+1
    if len(extended_tags) == 0:
        return store_data,""
    nxG,text_index = priority_based_structure(extended_tags,parent_node)
    if len(list(text_index.keys())) == 0:
        return store_data,nxG
    list_headings = []
    root_node = list(nxG.adj[parent_node])
    for r in root_node:
        # print(list(nxG.adj[r]))
        hl = list(nxG.adj[r])
        list_headings.extend(generate_heading(hl[1:],str(r),nxG))
    store_data["Topic_Name"].append(topicName)
    store_data["URL"].append(link)
    store_data["Text_Index"].append(text_index)
    store_data["Network"].append(list_headings)
    store_data["All_Tags"].append(extended_tags)
    store_data["nxG"].append(nxG)
    return store_data,nxG

def get_context(search_query,user_query):
    links = get_google_search_links(search_query)
    if len(links)==0:
        links = get_bing_search_links(search_query)
        if len(links) == 0:
            return "Internet down or max tries"
    store_data = {"Topic_Name":[],"URL":[],"All_Tags":[],"Text_Index":[],"Network":[],"nxG":[]}
    store_data,nxG = start_network(links,store_data,search_query)
    df = pd.DataFrame.from_dict(store_data)
    df.drop("All_Tags",axis = 1,inplace = True)
    print(df.head())
    keys = list(df["Text_Index"][0].keys())
    values = list(df["Text_Index"][0].values())
    df_new = pd.DataFrame()
    df_new["keys"] = keys
    df_new["values"] = [i.replace("\n","").replace("\t","") if i[:6]!="<code>" else i for i in values ]
    df_new["length"] = [len(i.split(" ")) for i in values]
    df_new = df_new.drop_duplicates(subset="values")
    filtered_df = df_new[df_new["length"] >=30]
    filtered_df = filtered_df[filtered_df["values"].str[:6] != "<code>"]
    filtered_df = filtered_df.drop_duplicates(subset="values")
    # The sentences to encode
    sentences = list(filtered_df["values"])
    # 2. Calculate embeddings by calling model.encode()
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    query = user_query
    query_embeddings = model.encode(query)
    print(query_embeddings.shape)
    # 3. Calculate the embedding similarities
    similarities = model.similarity(embeddings, query_embeddings)
    filtered_df["similarity"] = similarities.numpy().flatten().tolist()
    sorted_df = filtered_df.sort_values(by="similarity",ascending=False)
    sorted_df = sorted_df.head(10)
    sorted_keys = list(sorted_df["keys"])
    keys_track = []
    context_small = ""
    context_med = ""
    context_large = ""
    images = []
    for idx,key in enumerate(sorted_keys):
        try:    
            start = list(nxG.adj[key])[0]
            end = list(nxG.adj[start])[-1]
            if start in keys_track:
                continue
            keys_track.append(start)
            for i in range(start,end+1):
                try:
                    if "<image>" in df_new["values"][i] and "https" in df_new["values"][i] and df_new["values"][i][-3:] in ["png","jpg"]:
                        images.append(df_new["values"][i].replace("<image>",""))
                    context_large = context_large+df_new["values"][i]+"\n"
                    if idx <=3:
                        context_small = context_small+df_new["values"][i]+"\n"
                        context_med = context_med+df_new["values"][i]+"\n"
                    elif idx >=3 and idx <=5:
                        context_med = context_med+df_new["values"][i]+"\n"
                except Exception as e:
                    pass
        except Exception as e:
            print(e)

    if len(context_large.split(" ")) <3000:
        return context_large,images,nxG
    elif len(context_med.split(" ")) <3000:
        return context_med,images,nxG
    elif len(context_small.split(" ")) <3000:
        return context_small,images,nxG
    else:
        context = " ".join(context_small.split(" ")[:3000])
        print(images)
        return context,images,nxG
    