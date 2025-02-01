
from typing import List, Union, Any
from pydantic import validate_arguments, validate_call, BaseModel, Field
from functools import wraps

import json
import openai

#https://github.com/minimaxir/simpleaichat/blob/main/simpleaichat/utils.py
#this function is used to remove the title key from the schema as openai does not use it
#WATCH-OUT: do not use the "title" key in your own schema!!!!
def remove_a_key(d, remove_key):
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                remove_a_key(d[key], remove_key)

#this is a decorator that will create the openai schema for the function
class openai_tool:
    def __init__(self, func) -> None:
        self.func = func
        self.validate_func = validate_arguments(func)
        parameters = self.validate_func.model.schema()
        remove_a_key(parameters, "title")
        remove_a_key(parameters, "args")
        remove_a_key(parameters, "kwargs")
        remove_a_key(parameters, "v__duplicate_kwargs")
        remove_a_key(parameters, "special_variables")

        #remove_a_key(parameters, "additionalProperties")
        parameters['additionalProperties']=False
        self.openai_schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__,
                "strict":True,
                "parameters": parameters,
            }
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        @wraps(self.func)
        def wrapper(*args, **kwargs):
            return self.validate_func(*args, **kwargs)
        return wrapper(*args, **kwargs)

    def from_response(self, arguments: dict):
        """Execute the function from the response of an openai chat completion"""
        return self.validate_func(**arguments)


#use this function to create the openai schema of a class member function
def get_openai_schema(func):
    validate_func = validate_arguments(func) #deprecated,says it should be replaced by validate_call
    #validate_func = validate_call(func) #but this does not work, it faills in the next line
    parameters = validate_func.model.schema()
    remove_a_key(parameters, "title")
    remove_a_key(parameters, "args")
    remove_a_key(parameters, "kwargs")
    remove_a_key(parameters, "v__duplicate_kwargs")
    remove_a_key(parameters, "special_variables")
    parameters['additionalProperties']=False
    openai_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__,
            "strict":True,
            "parameters": parameters,
        }
    }
    return openai_schema

########
if __name__ == "__main__":

    @openai_tool
    def test(
        action:str = Field(description="this is an enum", enum=["a", "b"]),
        id:str = Field(description="this is a string"),
        ):
        """description of the function.
        """
        return action, id

    test_description = test.openai_schema
    #print(json.dumps(test_description,indent=4))

    v = {"action":"a", "id":"boo"}
    r = test(**v)
    #print(r)


    class Test(BaseModel):
        """description of the function"""
        action:str = Field(description="this is an enum", enum=["a", "b"]),
        id:str = Field(description="this is a string"),

    t = openai.pydantic_function_tool(Test)
    #print(json.dumps(t, indent=4))



    class S:
        def __init__(self, a):
            self.a = a
        
        def test(self, 
                action:str = Field(description="this is an enum", enum=["a", "b"]),
                id:str = Field(description="this is a string"),
            ):
            """description of the function."""
            return self.a, action, id


    s = S("hello")
    t = get_openai_schema(s.test)
    #print(json.dumps(t, indent=4))
    r = s.test(**v)
    #print(r)
