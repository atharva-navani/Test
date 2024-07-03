import os
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
#import spacy
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load English tokenizer, tagger, parser, and NER
nlp = spacy.load("en_core_web_lg")

def identify_chart(prompt):
    chart_keywords = {
        "Line Chart": {"keywords": ["trend", "time series", "change over time", "pattern", "fluctuation", "variation", "trajectory", "development", "evolution", "movement"], "weight": 1.5},
        "Bar Chart": {"keywords": ["comparison", "distribution", "category-wise analysis", "variation", "disparity", "discrepancy", "inequality", "diversity", "contrast", "difference"], "weight": 1.5},
        "Pie Chart": {"keywords": ["percentage", "proportion", "composition", "share", "distribution", "ratio", "part", "fraction", "portion", "segment"], "weight": 1.5}
    }

    doc = nlp(prompt.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop]

    matched_charts = {}
    for chart, chart_info in chart_keywords.items():
        weight = sum(1 for keyword in chart_info["keywords"] if keyword in tokens) * chart_info["weight"]
        if weight > 0:
            matched_charts[chart] = weight

    matched_charts = {k: v for k, v in sorted(matched_charts.items(), key=lambda item: item[1], reverse=True)}

    if not matched_charts:
        matched_charts["Pie Chart"] = 1.0

    return list(matched_charts.keys())

load_dotenv()
app = Flask(__name__)
CORS(app)

model_name = "gpt-3.5-turbo"
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name=model_name,
    api_key=openai_api_key,
    temperature=0.0
)

db = SQLDatabase.from_uri("mysql+pymysql://avnadmin:AVNS_nflzgxHyli00Kc1NeCh@economic-calendar-atharvan16-e10e.d.aivencloud.com:21347/customer")

write_query = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)

plain_text_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

json_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, provide a JSON object with the details needed for visualization.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer (in JSON format): """
)

plain_text_answer = plain_text_prompt | llm | StrOutputParser()
json_answer = json_prompt | llm | StrOutputParser()

chain_plain_text = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | plain_text_answer
)

chain_json = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | json_answer
)

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        plain_text_ans = chain_plain_text.invoke({"question": user_question})
        json_ans = chain_json.invoke({"question": user_question})

        try:
            json_data = json.loads(json_ans)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response", exc_info=True)
            return jsonify({"error": "Failed to parse response as JSON", "response": json_ans}), 500

        chart_type = identify_chart(user_question)

        return jsonify({
            "plain_text_answer": plain_text_ans,
            "json_answer": json_data,
            "chart_type": chart_type[0]
        })
    except Exception as e:
        logger.error("Error processing request", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)










# ## Improvedd 2


# import os
# import json
# from dotenv import load_dotenv
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.utilities import SQLDatabase
# from langchain.chains import create_sql_query_chain
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
# from operator import itemgetter
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# import spacy
# import logging

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load English tokenizer, tagger, parser, and NER
# nlp = spacy.load("en_core_web_lg")

# def identify_chart(prompt):
#     chart_keywords = {
#         "Line Chart": {"keywords": ["trend", "time series", "change over time", "pattern", "fluctuation", "variation", "trajectory", "development", "evolution", "movement"], "weight": 1.5},
#         "Bar Chart": {"keywords": ["comparison", "distribution", "category-wise analysis", "variation", "disparity", "discrepancy", "inequality", "diversity", "contrast", "difference"], "weight": 1.5},
#         "Pie Chart": {"keywords": ["percentage", "proportion", "composition", "share", "distribution", "ratio", "part", "fraction", "portion", "segment"], "weight": 1.5}
#     }

#     doc = nlp(prompt.lower())
#     tokens = [token.lemma_ for token in doc if not token.is_stop]

#     matched_charts = {}
#     for chart, chart_info in chart_keywords.items():
#         weight = sum(1 for keyword in chart_info["keywords"] if keyword in tokens) * chart_info["weight"]
#         if weight > 0:
#             matched_charts[chart] = weight

#     matched_charts = {k: v for k, v in sorted(matched_charts.items(), key=lambda item: item[1], reverse=True)}

#     if not matched_charts:
#         matched_charts["Pie Chart"] = 1.0

#     return list(matched_charts.keys())

# load_dotenv()
# app = Flask(__name__)
# CORS(app)

# model_name = "gpt-3.5-turbo"
# openai_api_key = os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(
#     model_name=model_name,
#     api_key=openai_api_key,
#     temperature=0.0
# )

# db = SQLDatabase.from_uri("mysql+pymysql://avnadmin:AVNS_nflzgxHyli00Kc1NeCh@economic-calendar-atharvan16-e10e.d.aivencloud.com:21347/customer")

# write_query = create_sql_query_chain(llm, db)
# execute_query = QuerySQLDataBaseTool(db=db)

# plain_text_prompt = PromptTemplate.from_template(
#     """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

# Question: {question}
# SQL Query: {query}
# SQL Result: {result}
# Answer: """
# )

# json_prompt = PromptTemplate.from_template(
#     """Given the following user question, corresponding SQL query, and SQL result, provide a JSON object with the details needed for visualization.

# Question: {question}
# SQL Query: {query}
# SQL Result: {result}
# Answer (in JSON format): """
# )

# plain_text_answer = plain_text_prompt | llm | StrOutputParser()
# json_answer = json_prompt | llm | StrOutputParser()

# chain_plain_text = (
#     RunnablePassthrough.assign(query=write_query).assign(
#         result=itemgetter("query") | execute_query
#     )
#     | plain_text_answer
# )

# chain_json = (
#     RunnablePassthrough.assign(query=write_query).assign(
#         result=itemgetter("query") | execute_query
#     )
#     | json_answer
# )

# @app.route('/ask', methods=['POST'])
# def ask():
#     user_question = request.json.get('question')
#     if not user_question:
#         return jsonify({"error": "No question provided"}), 400

#     try:
#         # Validate and sanitize the user question
#         if not isinstance(user_question, str) or len(user_question.strip()) == 0:
#             return jsonify({"error": "Invalid question provided"}), 400

#         # Generate and execute the SQL query
#         plain_text_ans = chain_plain_text.invoke({"question": user_question})
#         json_ans = chain_json.invoke({"question": user_question})

#         try:
#             json_data = json.loads(json_ans)
#         except json.JSONDecodeError:
#             logger.error("Failed to parse JSON response", exc_info=True)
#             json_data = {"error": "Failed to parse response as JSON", "response": json_ans}

#         chart_type = identify_chart(user_question)

#         return jsonify({
#             "plain_text_answer": plain_text_ans,
#             "json_answer": json_data,
#             "chart_type": chart_type[0]
#         })
#     except Exception as e:
#         logger.error("Error processing request", exc_info=True)
#         return jsonify({"error": "An error occurred while processing your request. Please try again later."}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
