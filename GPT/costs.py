class ModeloGPT:
    def __init__(self, name, context, price_promt, price_completion):
        self.name = name
        self.context = context
        self.price_promt = float(price_promt.strip('$'))
        self.price_completion = float(price_completion.strip('$'))


# Create instances of the class for the provided data
GPT3 = ModeloGPT("gpt-35-turbo", "4K", "$0.0015", "$0.002")
GPT3_BIG = ModeloGPT("gpt-35-turbo-16k", "16K", "$0.003", "$0.004")
GPT4 = ModeloGPT("gpt-4", "8K", "$0.03", "$0.06")
GPT4_BIG = ModeloGPT("gpt-4-32k", "32K", "$0.06", "$0.12")


if __name__ == "__main__":

    models = [GPT3, GPT3_BIG, GPT4, GPT4_BIG]

    for m in models:
        print("Model Name:", m.model_name)
        print("Tam contex in tokens:", m.context)
        print("Price input token:", m.price_promt)
        print("Price output:", m.price_completion)
        print("-"*40)