#!/usr/bin/python
import ir_system
import rocchio_algorithm
import ir_evaluator
import sys
import re
import os
import csv

def preprocess_userinput(user_input):

    if os.path.exists(user_input): 
        try:
            list_texts = re.split(
                r".I \d*\n.W\n", open(user_input).read())[1:]
            return list_texts
        except IOError:
            print(user_input + " - Archivo no encontrado")
            sys.exit(0)
    else:
        # the user has provided a query or a text
        only_query_id = input("Escriba el ID de la Consulta introducida:\n")
        return user_input, only_query_id

def create_ir_system(irmodel_choice, corpus, query):
    if irmodel_choice == 0:
        return ir_system.IRBoolean(corpus, query)
    elif irmodel_choice == 1:
        return ir_system.IR_tf(corpus, query)
    elif irmodel_choice == 2:
        return ir_system.IR_tf_idf(corpus, query)


def execute_IRsystem_prompt(corpus_text, query_text, only_query_id):

    print("\n Modelos Disponibles: \n 0:Booleano\n 1:TF\n 2:TF-IDF\n 3:LDA\n 4:LDA Multicore\n 5:LSI\n 6:RP\n 7:LogEntropyModel\n \n")
    irmodel_choice = input(
        "Por Favor, Seleccione el Modelo indicando el número que lo representa:\n")

    ir = create_ir_system(int(irmodel_choice), corpus_text, query_text)

    irevaluator_choice = input(
        "Quieres evaluar el rendimiento del modelo seleccionado (si/no)? \n")

    if(irevaluator_choice == "si"):
        relevances_input = input(
            "Introduzca la ruta del documento que contiene las relevancias:\n")
        with open(relevances_input, 'r+') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            relevances = []
            for row in spamreader:
                relevances.append(row)

        ir_evaluator.IREvaluator(
            relevances, ir.ranking_query, True, only_query_id)

    continue_choice = input(
        "Quieres probar otro modelo (si/no)? \n")

    if(continue_choice == "si"):
        execute_IRsystem_prompt(corpus_text, query_text, only_query_id)

    return ir

#################################################################################
# @brief   execute_Rocchio_prompt
#  @details This method is used to interact with the user to execute the rocchio
#           algorithm evaluation
#################################################################################


def execute_Rocchio_prompt(query_text, corpus_text, ir, only_query_id):
    rocchio_choice = input(
        "Quieres Ejecutar el algoritmo de Optimización de Rocchio (si/no)? \n")
    if(rocchio_choice == "si"):
        print("------------Ejecutando el Algoritmo de Rocchio------------")
        
        user_improvement = input(
            "Please, choose the X (e.g. X=20) first documents in the ranking and marks them as being relevant or non relevant according to the relevance assessments in MED.REL  \n")

        rankings = [list(i) for i in ir.ranking_query[1]]  # convert to a list
        pos = 0
        while pos < int(user_improvement):
            answer = input("Es relevante el documento con ID " +
                               str(rankings[pos][0]) + " (S/N)?")
            if (answer == 's') or (answer == 'S'):
                rankings[pos][1] = 1
            pos += 1
        # 5) According 
        rocchio = rocchio_algorithm.RocchioAlgorithm(
            query_text, corpus_text, rankings, ir)
        # 6) The system launchs the new query and presents a new ranking.
        # 7) A new P/R curve is generated and compared to the previous one.
        answer = 's'
        while ((answer == 's') or (answer == 'S')):
            ir = execute_IRsystem_prompt(
                corpus_text, rocchio.new_query, only_query_id)
            # desired recall and precision to be chosen by the user
            answer = input(
                "Quieres Ejecutar nuevamente el algoritmo de Optimización de Rocchio (si/no)?")
    return


if __name__ == '__main__':
   corpus_input = input("Introduce la ruta del dataset:\n")
   corpus_text = preprocess_userinput(corpus_input)

   query_input = input(
      "Introduce una consulta:\n")
   query_text, only_query_id = preprocess_userinput(query_input)

   ir = execute_IRsystem_prompt(corpus_text, query_text, only_query_id)

   execute_Rocchio_prompt(
        query_text, corpus_text, ir, only_query_id)
