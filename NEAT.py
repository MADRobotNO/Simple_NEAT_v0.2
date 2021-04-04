from Model import Model
import numpy as np
import copy


class Neat:

    def __init__(self, number_of_inputs=2, number_of_outputs=1, number_of_models=1, mutation_rate=0.1):
        self.list_of_innovations = []
        self.list_of_all_models = []
        self.mutation_rate = mutation_rate
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.number_of_models = number_of_models

        parent_model = self.initialize_model()
        self.generate_population(parent_model)

    def generate_population(self, parent_model, parent_model_two=None):
        for i in range(len(self.list_of_all_models), self.number_of_models):
            model = self.model_from_parent_structure(parent_model)
            model.mutate(self.list_of_innovations, parent_model, parent_model_two)
            self.list_of_all_models.append(model)

    def model_from_parent_structure(self, parent_model):
        model = copy.deepcopy(parent_model)
        model.model_id = len(self.list_of_all_models)
        model.outputs = None
        model.score = 0.0
        model.regenerate_random_weights_bias()
        return model

    def initialize_model(self):
        model = Model(len(self.list_of_all_models), self.number_of_inputs, self.number_of_outputs, self.mutation_rate,
                      self.list_of_innovations)
        self.list_of_all_models.append(model)
        return model

    def fit(self, input_data, number_of_generations):
        for generation in range(1, number_of_generations+1):
            print("Generation:", generation)
            for model in self.list_of_all_models:
                model.mutate(self.list_of_innovations)
                # if len(model.list_of_all_layers) > 3:
                print(model)

    def __str__(self):
        return_string = "Current NEAT state:\n"
        for model in self.list_of_all_models:
            return_string += model.__str__() + "\n"
        return_string += "List of innovations:\n" + str(self.list_of_innovations)
        return return_string
