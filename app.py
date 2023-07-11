from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms import OpenAI
import openai
import streamlit as st

### AWS Libraries
from requests_aws4auth import AWS4Auth
import boto3
from boto3.dynamodb.conditions import Key, Attr

import json
import requests
import pandas as pd
from typing import Optional, Type

### Configurations
region = 'us-east-1' 
service = 'es'
session = boto3.Session()
sts = session.client("sts")
# Assume the role that owns OpenSearch Domain
response = sts.assume_role(
    RoleArn="arn:aws:iam::494897422457:role/service-role/lambda_dynamodb",
    RoleSessionName="run-opensearch"
)
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
headers = { "Content-Type": "application/json" }

# The OpenSearch domain endpoint
host = 'https://search-aderas-gabs-7wmy5nh5ue564vpv63uogepqbe.us-east-1.es.amazonaws.com' 

# OpenAI Credentials
openai_api_key = st.secrets["openai_api_key"]

llm = ChatOpenAI(model="gpt-4", temperature=0)

class OpenSearchKNN(BaseTool):
    name = "OpenSearchKNN"
    description = """
    This tool is useful for when you need to answer questions about government contracts by searching 
    using semantic embeddings/KNN. This tool will only return the 20 most representative records.
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        
        response = openai.Embedding.create(
          input=query,
          model="text-embedding-ada-002"
        )
        
        print(query)

        question_vectors = response['data'][0]['embedding']

        index = 'usaspending_sql'
        url = host + '/' + index + "/_search"

        # You can search for a specific year or run a query, but not both
        # myFY = '2022'
        myquerystr = 'FiscalYear: (2022 OR 2023)'

        knnquery = {
          "size": 20,
          "query": {
            "bool": {
              "filter": {
                "bool": {
                  "must": [
                #             {
                #               "match": {
                #                 "FiscalYear": myFY
                #                 }
                #             }    
                    {
                      "query_string": {
                         "query": myquerystr
                        }
                    }
                  ]
                }
              },
              "must": [
                {
                  "knn": {
                    "vectors": {
                      "vector": question_vectors,
                      "k": 20
                    }
                  }
                }
              ]
            }
          }
        }

        r = requests.get(url, auth=awsauth, headers=headers, data=json.dumps(knnquery))
        # print(r.text)
        segments = json.loads(r.text)

        context = ""
        recnbr = 1
        for i in segments['hits']['hits']:
            context = "Record #"+str(recnbr)+"\n"
            # print('-'*60)
            # print(i['_source']['text'])
            context = context + i['_source']['text'] + "\n-------\n"
            recnbr += 1

        knnprompt=f"""Based upon the following text provide a detailed answer to the question: "{query}"

        Text: {context}
        """
        response = openai.ChatCompletion.create(
            messages = [{"role": "user", "content": knnprompt}],
            temperature=0,
            max_tokens=3000,
            frequency_penalty=0,
            presence_penalty=0,
            model='gpt-4')
#             model='gpt-3.5-turbo-16k')

        # print("Question: "+question+"\n")
        return response.choices[0].message.content
        
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("The USASpendingKNN tool does not support async")


class OpenSearchSQL(BaseTool):
    name = "OpenSearchSQL"
    description = """
    This tool is useful for when you need to answer questions about government contracts by submitting 
    an ANSI SQL query against the contract database. This tool only accepts valid SQL queries.
    The output is in .CSV format. This tool is capable of returning all the records in the database.
    When comparing columns to string values, use a 'LIKE' statement rather than an "=" statement.
    The FiscalYear is an integer value stored as 'YYYY'. 
    When you write a SQL statement always begin with SELECT and end with ";".
    Do not format your SQL statement with line feeds.
      
    The table and fields include:
    
    usaspending_sql (ContractID, AgencyName, SubAgencyName, AwardAmount, PotentialAwardAmount, AwardDate, FiscalYear": int(csvname[2:6]), StartDate, PotentialEndDate, Recipient, RecipientAddress, PlaceOfPerformance, ContractType, Setaside, NAICS", NAICSDescription, NumberOfOffers, Description, ProductCode) 
        
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        # The OpenSearch domain endpoint
        url = host + "/_plugins/_sql?format=csv"        
        
        prefix1 = "```sql"
        prefix2 = "```"
        
        if query.startswith(prefix1):
            newquery = query[7:-4]
        elif query.startswith(prefix2):
            newquery = query[4:-4]
        else: 
            newquery = query
        print("newquery:", newquery)
        myquery = {
          "query": newquery
        }

        r = requests.post(url, auth=awsauth, headers=headers, data=json.dumps(myquery))
        return r.text  
                
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


DuckDuckGoSearchRun(name="Search")

tools = [OpenSearchSQL(), OpenSearchKNN(),DuckDuckGoSearchRun(name="Search")]
search_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    return_intermediate_steps=True,
)

st.set_page_config(page_title="USASpending GABS", layout="wide", initial_sidebar_state="collapsed")
st.title("USASpending GABS")

with st.sidebar:
    model_radio = st.radio(
        "Select ChatGPT Model",
        ("gpt-3.5-turbo", "gpt-4", "gpt-3.5-new")
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="List the 10 largest projects referencing Oracle in the description field."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    myprompt = f"""You are a researcher tasked with finding detailed answers to the users question. 
    Be thorough in your searching.

    Question: {prompt}
    """
    with st.spinner("Processing..."):
        response = search_agent(myprompt)

    st.session_state.messages.append({"role": "assistant", "content": response['output']})    
    st.chat_message('assistant').write(response['output'])
    with st.expander('Details', expanded=False):
        st.write(response["intermediate_steps"])
