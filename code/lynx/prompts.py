'''
prompt vs. prefix
- prompt is inserted as user prompt to the model-specific prompt template.
- prefix is right-concatenated to the filled-in template and is used to control how the machine answer should start with.
'''

prompt = "Do not say 'it depends' and provide a single answer. Does traffic have to stop?"
prefix = "the"
