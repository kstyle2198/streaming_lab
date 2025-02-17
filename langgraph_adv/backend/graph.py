from typing import Annotated, TypedDict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_groq import ChatGroq

# Define FastAPI app
app = FastAPI()

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list, add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str




graph = StateGraph(GraphsState)

# Core invocation of the model
def _call_model(state: GraphsState):
    messages = state["messages"]
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0, streaming=True)
    response = llm.invoke(messages)
    return {"messages": [response]}  # add the response to the messages using LangGraph reducer paradigm

# Define the structure (nodes and directional edges between nodes) of the graph
graph.add_edge(START, "modelNode")
graph.add_node("modelNode", _call_model)
graph.add_edge("modelNode", END)

# Compile the state graph into a runnable object
graph_runnable = graph.compile()

def invoke_our_graph(st_messages, callables):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return graph_runnable.invoke({"messages": st_messages}, config={"callbacks": callables})

# Define Pydantic model for request body
class RequestModel(BaseModel):
    messages: List

@app.post("/invoke")
def invoke_graph(request: RequestModel):
    try:
        result = invoke_our_graph(request.messages, [])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
