import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI

# df = pd.read_csv("./sales_data_sample.csv", encoding='Latin-1')
users_df = pd.read_csv('Users.csv')
recipients_df = pd.read_csv('Recipients.csv')
items_df = pd.read_csv('Items.csv')
clicks_df = pd.read_csv('Clicks.csv')
purchases_df = pd.read_csv('Purchases.csv')

prefix = "Save plots and visualizations that you create using plt.savefig() or any other relevant save function. After saving, just say here is the ___ graph. You must provide insightful recommendations and actionable insights for the prompts."

# suffix = "You must provide insightful recommendations and actionable insights for the prompts."

agent = create_pandas_dataframe_agent(OpenAI(temperature=0), clicks_df, prefix=prefix, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

prompt = input("How can I help you today?\n")

print(agent.invoke(prompt))

# Show me the relationship between sales of a product and it's pricing, and suggest the optimal pricing strategy accordingly. Provide any visualizations if necessary