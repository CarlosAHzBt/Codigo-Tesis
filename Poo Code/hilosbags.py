from BagFile import BagFile
import os
import threading

class ProcesadorBagsHilos:
    def __init__(self, bag_files_path):
        self.bag_files_path = bag_files_path
        self.bag_files = self.get_bag_files()
    def get_bag_files(self):
        """
        Obtiene la lista de archivos .bag en el directorio especificado.
        """
        return [f for f in os.listdir(self.bag_files_path) if f.endswith('.bag')]
    
    def process_bag_file_thread(self, bag_file_path):
        """
        Método para procesar un archivo .bag en un hilo separado.
        """
        bag = BagFile(bag_file_path, "ProcesamientoDeBags")
        bag.process_bag_file()

    def process_bag_files_concurrently(self):
        """
        Procesa todos los archivos .bag en el directorio especificado utilizando múltiples hilos.
        """
        threads = []
        for bag_file in self.bag_files:
            bag_file_path = os.path.join(self.bag_files_path, bag_file)
            thread = threading.Thread(target=self.process_bag_file_thread, args=(bag_file_path,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

# Ejemplo de uso
procesador = ProcesadorBagsHilos('bags')
procesador.process_bag_files_concurrently()
