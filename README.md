<div align="center">
 <img width="100" align="center" src="https://github.com/Lorylan/FORxREN/assets/63661809/1b6df4a3-9563-4f28-90d4-f2e9a20b749c"></img>
 <h1 >Tesina de Licenciatura en Informática </h1>
</div>

Este repositorio contiene la tesina de licenciatura en informática, titulada "Inteligencia Artificial Explicable: Técnicas de extracción de reglas en redes neuronales artificiales".

## Autores

- [Milagros Aylén Jacinto](https://github.com/Lorylan)
- [Martín Moschettoni](https://github.com/Nettaros)

## Directoras
- Claudia Pons
- Gabriela Pérez

## Resumen
Las redes neuronales artificiales tienen la capacidad de alcanzar altos niveles de precisión en tareas de clasificación, pero la imposibilidad de comprender y validar el proceso de decisión de un sistema de IA es un claro inconveniente. 
En esta tesina se diseñó y desarrollo un algoritmo de extracción de reglas llamada FORxREN. Se orienta hacia la extracción de reglas fieles y fácilmente interpretables que arrojen luz sobre el razonamiento subyacente de la red neuronal con la que fue entrenada. Este enfoque permitió analizar, comprender su utilidad y realizar modificaciones dirigidas a mejorar la explicabilidad de las redes neuronales.

## Trabajo Realizado
Se analizaron distintas técnicas de extracción de reglas, incluyendo [ECLAIRE](https://arxiv.org/abs/2111.12628), [FERNN](https://link.springer.com/article/10.1023/A:1008307919726) y [DeepRED](https://link.springer.com/chapter/10.1007/978-3-319-46307-0_29), cada una con un enfoque particular. Posteriormente, mediante ingeniería inversa del método de [RxREN](https://link.springer.com/article/10.1007/s11063-011-9207-8) y ajustando su enfoque original para dar prioridad a la fidelidad de las reglas sobre la precisión, se desarrolló el algoritmo [FORxREN](https://publicaciones.sadio.org.ar/index.php/JAIIO/article/view/551). Este algoritmo fue evaluado utilizando los conjuntos de datos Iris, WBC y Wine, mediante la construcción de múltiples redes neuronales artificiales.

## Conclusiones 
Los resultados de las ejecuciones han demostrado que FORxREN genera reglas con un alto porcentaje de fidelidad. Además, se ha observado que, fue posible aumentar la generalidad de las reglas (y, por lo tanto, su comprensibilidad) sin perder la fidelidad.
Ha quedado demostrado que dicha orientacion a la fidelidad por parte de FORxREN puede ser una estrategia efectiva para mejorar la explicabilidad de la red y generar reglas más fiables en determinadas aplicaciones. Además de que la explicabilidad en redes
neuronales artificiales es de suma importancia en la actualidad.

## Trabajos Futuros
- Investigar el método propuesto en redes neuronales complejas y de mayor escala para evaluar 
su escalabilidad y rendimiento.
- Realizar estudios empíricos para evaluar su eficacia en diferentes dominios y tareas, incluyendo la clasificación de datos no numéricos.
- Examinar el impacto de diferentes arquitecturas de red en la explicabilidad, y se investigará su potencial en ombinación con otras técnicas de explicabilidad.
- Evaluar la comprensibilidad de manera formal.

## Estructura del Repositorio

- `{nombreDataset}_model/`: Carpeta con los datos de la red correspondiente al dataset indicado en el nombre.
- `draw_nn.py`: Archivo que genera los gráficos de las arquitecturas de las redes neuronales.
- `FORxREN.py`: Archivo que contiene el algoritmo FORxREN.
- `main.py`: Archivo que contruye/carga los modelos de las redes neuronales para luego ejecutar el algoritmo FORxREN. 
- `configuration.py`: Archivo que contiene la configuracion de ejecución.

## Configuración

> [!NOTE]
> Si desea trabajar con otro dataset, tiene que agregar la configuración del mismo en el archivo `main.py`
- `DATASET `: Por defecto iris - Opciones: "iris", "wbc", "wine"
- `EXECUTION_MODE` : Por defecto 1 (Algoritmo completo) - Opciones: 1, 2 (Sin actualización de reglas), 3 (Solo reglas iniciales, no realiza pruning ni actialización de reglas)
- `TEST_PERCENT`: Por defecto 0.2 - Porcentaje utilizado para el test, 0.2 = 20%
- `MAX_FIDELITY_LOSS`: Por defecto 0.05 - Porcentaje que indica el máximo de fidelidad que se acepta perder apagando neuronas. 
- `CREATE_NN`: Por defecto False - Booleano que indica si queres creear una nueva red neuronal o usar una de las redes guardadas en el repositorio.  
- `SAVE_NN` = Por defecto False - Booleano que indica si queres guardar la red o no. 
- `SHOW_STEPS` = Por defecto True - Booleano que indica si queres que te muestre en la consola el paso a paso de lo que va haciendo el algoritmo.

## Requisitos
- Para visualizar el documento PDF de la tesis, solo necesitas un visor de PDF estándar.
- Para ejecutar el codigo necesita tener intalado [Python](https://www.python.org/downloads/)

## Modo de uso
> [!IMPORTANT]
> Si es la primera vez que va a ejecutar el código se recomiendo utilizan el siguiente comando primero para instalar las dependencias: 
> ```pip install -r requirements.txt```

```
python .\FORxREN\experiments.py
```


## Licencia

El contenido de este repositorio está bajo [Licencia MIT](https://opensource.org/licenses/MIT).

## Contacto

Para cualquier pregunta o comentario relacionado con este trabajo, no dudes en contactarnos por correo electrónico a [mili.aylen.j@gmail.com](mili.aylen.j@gmail.com) o [moschettonimartin@gmail.com](moschettonimartin@gmail.com).
