# proyecto_metodos

Antes de hacer clone del proyecto ejecutar lo siguiente para el control de libreias:
pip install pip-tools

Clonar el repositorio

Ya que tengan el proyecto abierto, hacen un entorno virtual:
python -m venv venv

Y activan el entorno:
venv\Scripts\activate

Hacen la instalación de librerías:
pip-sync

Ya que tengan todo descargado, pueden correr el programa con lo siguiente:
streamlit run app.py


Nota: Si van a agregar nuevas librerias, no las instalen directamente, pongan el nombre de la librería en requirements.in y luego corren lo siguiente:
pip-compile

Ya que termine de compilar, ahor si le dan:
pip-sync

y cuando termine de instalar, debería de agregarse en requirements.txt y pues quiere decir que si se instalo en el entorno 
