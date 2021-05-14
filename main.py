from ir_system import IrSystem
import pickle

def main_exec(irsystem):
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
            
            irsystem.executeRochio(query_id, relevants, 1, 0.9, 0.5)

            print("\n---------- Algoritmo de Rochio Realizado Correctamente -----------\n")

            irsystem.search(query_id = query_id, preview=250)
            irsystem.evaluate_query(query_id, True)

            with open('irsystem.backup', 'wb') as data:    
                pickle.dump(irsystem, data)

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

if __name__ == '__main__':
    dataset = input('Elige un Dataset: \n1 - Cranfield \n2 - MED \nEnter - Para terminar\n-> ')
    if dataset == '1' or dataset == '2':
        backup = input("Cargar Backup: (si/no) \n-> ")
        if backup == 'si':
            try:
                with open(f'irsystem.backup{dataset}', 'rb') as data:    
                    irsystem = pickle.load(data)

            except FileNotFoundError:
                print("No hay ningun dataset cargado actualmente. Generando nuevamente el sistema...")
                irsystem = IrSystem(0.3, dataset)
        else:
            irsystem = IrSystem(0.3, dataset)

        with open(f'irsystem.backup{dataset}', 'wb') as data:    
            pickle.dump(irsystem, data)

        main_exec(irsystem)