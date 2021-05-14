from ir_system import IrSystem

if __name__ == '__main__':
    dataset = input('Elige un Dataset: \n1 - Cranfield \n2 - MED \nEnter - Para terminar\n-> ')
    if dataset == '1' or dataset == '2':
        irsystem = IrSystem(0.3, dataset)
    
        while True:
            print('\nOpciones:')
            mode = input(f"1 - Hacer una Consulta \n2 - Aplicar Retroalimentación de Rocchio a una Consulta \n3 - Analizar Rendimiento del Sistema \nEnter - Para terminar\n-> ")
            if mode == '1':
                query = input("\nEscribe una consulta: ")
                alpha = input("Escribe la Constante de Suavizado: ")
                irsystem.search(query, alpha)
            elif mode == '2':
                print('\nAplicar Retroalimentación de Rocchio a:')
                ask = [f'{query[0]} - {irsystem.querys[query[0]]["text"]}\n' for query in irsystem.searched.items()]
                query_id = input("".join(ask) + 'Elegir ID -> ')

                print("\n---------- Documentos Recuperados -----------\n")

                irsystem.search(query_id = query_id, preview=250)

                relevants = input("Seleccione el ID de los documentos que le parecen relevantes separados por espacios: \n->").split(' ')
                
                irsystem.executeRochio(query_id, relevants, 1, 0.75, 0.15)

                print("\n---------- Algoritmo de Rochio Realizado Correctamente -----------\n")

                irsystem.search(query_id = query_id, preview=250)

            elif mode == '3':
                print("\n---------- Análisis del SRI -----------\n")
                while True:
                    mode = input(f"1 - Análisis General \n2 - Análisis de una Consulta \nEnter - Atrás\n-> ")
                    if mode == '1':
                        irsystem.evaluate_system()
                    elif mode == '2':
                        print('\nOpciones:')
                        ask = [f'{query[0]} - {irsystem.querys[query[0]]["text"]}\n' for query in irsystem.searched.items()]
                        query = input("".join(ask) + 'Elegir ID -> ')
                        irsystem.evaluate_query(query, True)
                    else:
                        break
            else:
                break
