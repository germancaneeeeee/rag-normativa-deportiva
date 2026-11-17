# RAG local sobre normativa nacional del deporte

**Trabajo Integrador 2 — Procesamiento de Lenguaje Natural**
Tecnicatura Superior en Ciencias de Datos e Inteligencia Artificial — IFTS N.º 24
Autor: Germán Miguel Cané · Noviembre de 2026

Sistema de preguntas y respuestas que busca evidencia en tres leyes argentinas antes de responder, muestra los fragmentos que recibió el modelo y funciona completamente en la computadora local con Ollama.

---

## 1. Problema y destinatario

**Este sistema permite que dirigentes de clubes y federaciones deportivas consulten dudas sobre la normativa nacional del deporte en tres leyes argentinas, para encontrar rápido qué dice la ley y en qué artículo.**

**Puede responder** preguntas cuyo dato está escrito en las leyes del corpus, por ejemplo:
- qué objetivos tiene el Estado respecto del deporte;
- qué obligaciones tienen las federaciones deportivas en materia antidopaje;
- qué se considera un club de barrio y de pueblo y qué requisitos tiene para inscribirse.

**No puede responder:**
- datos que no están en estas leyes (cuotas, resultados deportivos, reglamentos de una federación, otras normas);
- interpretaciones jurídicas o casos concretos. **No reemplaza el asesoramiento legal**: muestra el fragmento de la ley para que la persona lo verifique.

---

## 2. Corpus

Los tres documentos están versionados en `datos/`. Se descargaron el **7/11/2026** desde argentina.gob.ar (Información Legislativa) y se guardaron desde el navegador con «Imprimir → Guardar como PDF».

| Archivo en `datos/` | Norma | Autoría | Sanción · Boletín Oficial | Versión | Origen | Págs. | Condiciones de uso |
|---|---|---|---|---|---|---:|---|
| `ley_20655_deporte.pdf` | Ley 20.655, Ley del Deporte | Honorable Congreso de la Nación Argentina | 21/03/1974 · 08/04/1974 | Texto actualizado | [argentina.gob.ar](https://www.argentina.gob.ar/normativa/nacional/ley-20655-27274/actualizacion) | 19 | Texto normativo oficial de acceso público; se cita la fuente |
| `ley_26912_dopaje.pdf` | Ley 26.912, Régimen Jurídico para la Prevención y el Control del Dopaje en el Deporte | Honorable Congreso de la Nación Argentina | 13/11/2013 · 26/12/2013 | Texto actualizado | [argentina.gob.ar](https://www.argentina.gob.ar/normativa/nacional/ley-26912-224421/actualizacion) | 76 | Texto normativo oficial de acceso público; se cita la fuente |
| `ley_27098_clubes_barrio.pdf` | Ley 27.098, Régimen de Promoción de los Clubes de Barrio y de Pueblo | Honorable Congreso de la Nación Argentina | 17/12/2014 · 22/01/2015 | Texto original | [argentina.gob.ar](https://www.argentina.gob.ar/normativa/nacional/norma-241139/texto) | 5 | Texto normativo oficial de acceso público; se cita la fuente |

Observaciones sobre los archivos:
- Los tres PDF tienen texto extraíble (no están escaneados), así que no hizo falta OCR.
- La Ley 26.912 aporta cerca del 78 % del texto del corpus.
- El texto oficial descargado tiene algunas erratas de origen que se dejaron tal cual (por ejemplo, «millas deportivas» en el art. 2 de la Ley 26.912 y «Registrada bajo el N° 26.655» al final de la Ley 20.655).

---

## 3. Arquitectura

```
datos/*.pdf  (3 leyes versionadas en GitHub)
    → PyPDFLoader: una página por Document, con metadatos source y page
    → limpieza: se quitan pies, URL, enlaces e íconos que agregó el navegador
    → RecursiveCharacterTextSplitter (1100 caracteres, solapamiento 180)
    → OllamaEmbeddings (bge-m3) → Chroma (chroma_db_normativa/)
pregunta → embedding con bge-m3 → los 3 fragmentos más cercanos (k = 3)
    → contexto con archivo y página → ChatPromptTemplate → ChatOllama (granite4)
    → Gradio: respuesta + fragmentos enviados al modelo, por separado
```

| Pieza | Herramienta y versión usada |
|---|---|
| Python | 3.12.14 (entorno creado con `uv`) |
| Modelo de embeddings | `bge-m3:latest` en Ollama (ID 790764642607, 1,2 GB) · vectores de 1024 dimensiones |
| Modelo generador | `granite4:latest` en Ollama (ID 4235724a127c, 2,1 GB) · `temperature=0` |
| Carga de PDF | `PyPDFLoader` (langchain-community) con pypdf 7.0.0 |
| Fragmentación | `RecursiveCharacterTextSplitter` (langchain-text-splitters 1.1.3) |
| Base vectorial | Chroma (langchain-chroma 1.1.0) |
| Interfaz | Gradio 6.31.0 |

Archivos del proyecto:

| Archivo | Contenido |
|---|---|
| `rag_normativa_deportiva.ipynb` | Recorrido completo y comentado: carga, limpieza, medición, comparación de cortes, índice, recuperación, generación, pruebas e interfaz |
| `app.py` | La misma aplicación, con los mismos pasos y parámetros, para ejecutar fuera del cuaderno |
| `resultados_pruebas.csv` | Salida de las cinco pruebas (fragmentos recuperados y respuesta del modelo) |
| `requirements.txt` | Paquetes de Python |
| `datos/` | Los tres documentos del corpus |

---

## 4. Instalación local

### Requisitos
- [Ollama](https://ollama.com) instalado y abierto.
- [uv](https://docs.astral.sh/uv/) instalado.
- Python 3.12 (uv lo descarga si hace falta).
- Unos 4 GB libres para los modelos.

> **Importante en Windows:** no trabajes dentro de una carpeta sincronizada (OneDrive, Google Drive, Dropbox). El entorno `.venv` tiene miles de archivos y la sincronización hace fallar la instalación (`os error 396`). Usá, por ejemplo, `C:\Users\<usuario>\TP_PLN\`.

### Pasos

1. Descargá o cloná el repositorio y entrá a su carpeta.

2. Descargá los dos modelos de Ollama:
   ```
   ollama pull granite4:latest
   ollama pull bge-m3
   ```
   Comprobá con `ollama list` que aparezcan los dos.

3. Creá el entorno e instalá los paquetes:

   | Paso | Windows (PowerShell) | Mac o Linux |
   |---|---|---|
   | Crear el entorno | `uv venv --python 3.12` | `uv venv --python 3.12` |
   | Permitir la activación (sólo en esa terminal) | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` | — |
   | Activarlo | `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
   | Instalar paquetes | `uv pip install -r requirements.txt` | `uv pip install -r requirements.txt` |

   Si la instalación falla con un error de *hardlink*, repetila copiando los archivos: `uv pip install -r requirements.txt --link-mode=copy`.

4. **Abrir la interfaz:** con Ollama abierto y el entorno activado, ejecutá
   ```
   python app.py
   ```
   La primera vez construye el índice (puede tardar varios minutos). Después abrí en el navegador la dirección que muestra la terminal (normalmente `http://127.0.0.1:7860`).

5. **Recorrer el cuaderno:** abrí la carpeta en VS Code, abrí `rag_normativa_deportiva.ipynb`, elegí `.venv` como kernel (Seleccionar kernel → Entornos de Python → `.venv`) y ejecutá las celdas en orden.

---

## 5. Reindexación

- El índice se guarda en **`chroma_db_normativa/`**. Esa carpeta **no se publica** (está en `.gitignore`): se reconstruye desde `datos/`.
- `app.py` construye el índice sólo si la colección está vacía. Para reconstruirlo, cerrá la aplicación, **borrá la carpeta `chroma_db_normativa/`** y volvé a ejecutar `python app.py`.
- El cuaderno, en la sección 6, **borra y reconstruye** su colección cada vez que se ejecuta esa celda.
- El nombre de la colección incluye el modelo de embeddings y el corte (`normativa_deporte_bge_m3_1100_180`). Si cambiás el corpus, el modelo, el tamaño o el solapamiento, se crea otra colección: así no se mezclan vectores de configuraciones distintas. Si cambiás los parámetros, cambialos igual en el cuaderno y en `app.py`.
- Con esta configuración el índice tiene **328 fragmentos**. Con otra versión de pypdf el número puede variar levemente, porque cambia la extracción del texto.
- No se publican `.venv/`, `.env` ni la base generada.

---

## 6. Pruebas

Los cinco casos se escribieron **antes** de evaluar el sistema, leyendo las leyes. Se ejecutaron en la sección 9 del cuaderno (salida en `resultados_pruebas.csv`) y se revisaron en la interfaz comparando cada fragmento con el PDF original.

| Pregunta | Respuesta / fuente esperada | Fragmentos recuperados (archivo y página) | Respuesta del modelo | ¿Qué observé? |
|---|---|---|---|---|
| ¿Cuáles son los objetivos fundamentales del Estado respecto del deporte según la Ley del Deporte? | Ley 20.655, art. 1, pág. 2 (incisos a a i) | Los 3: `ley_20655_deporte.pdf`, pág. 2 (art. 1, partido en 3 fragmentos) | Enumera los incisos a) a d) correctamente; omite e) a i) | Recuperación correcta. **Falla de generación**: respuesta incompleta, mezcla el final del inciso e) con el d) y cita sólo [1] aunque usa texto de [3] |
| ¿Qué obligaciones tienen las federaciones deportivas nacionales respecto de las normas antidopaje? | Ley 26.912, art. 2, pág. 2 (al probar apareció también el art. 82, págs. 42–43) | [1] y [2] `ley_26912_dopaje.pdf`, pág. 2 (art. 2) · [3] `ley_26912_dopaje.pdf`, pág. 42 (art. 82, incisos a–c) | Copia textualmente los incisos a), b) y c) del art. 82 | Recuperación buena (encontró el art. 82, más directo que el esperado). **Falla de corte**: los incisos d) a f) están en la pág. 43 y no llegaron. **Falla de generación**: no usó [1] y [2] y los citó igual |
| ¿Qué se considera un club de barrio y de pueblo? | Ley 27.098, art. 2, pág. 2 | [1] `ley_27098_clubes_barrio.pdf`, pág. 2 (art. 2 completo) · [2] pág. 2 (arts. 4–6) · [3] pág. 3 (arts. 6–8) | Reproduce textualmente la definición y cita [1] | **Correcto.** El artículo entra entero en un fragmento y quedó mucho más cerca que el resto (distancia 0,646 contra 0,838) |
| ¿Cuánto cuesta la cuota social de un club de barrio? | No figura en el corpus | [1] `ley_27098_clubes_barrio.pdf`, pág. 4 · [2] y [3] pág. 5 (multas, tarifa social, embargos) | «No encuentro esa información en los documentos.» | **Abstención correcta.** Chroma igual devolvió vecinos del mismo tema, sin el dato |
| ¿Quién ganó el último Campeonato Argentino de tiro con arco? | No figura en el corpus | [1] `ley_27098_clubes_barrio.pdf`, pág. 5 (cierre) · [2] `ley_20655_deporte.pdf`, pág. 19 (cierre) · [3] `ley_20655_deporte.pdf`, pág. 9 | «No encuentro esa información en los documentos.» | **Abstención correcta.** Recuperó fragmentos sin relación (cierres de las leyes) |

**Hallazgos**
- La búsqueda encontró el artículo correcto en las tres preguntas respondibles.
- En las dos preguntas sin respuesta el modelo se abstuvo. Esto **no garantiza** que se abstenga siempre: es un pedido del prompt, no una regla del sistema.
- Chroma siempre devuelve `k` fragmentos, aunque ninguno tenga el dato. En estas pruebas, las distancias de las preguntas sin respuesta (0,94 a 1,26) fueron más altas que las de las respondibles (0,51 a 0,65), pero cinco casos no alcanzan para fijar un umbral.
- Estas pruebas muestran el mecanismo; **no miden la precisión general** del sistema.

**Limitación observada.** Con artículos largos la respuesta sale incompleta. El caso más claro es el art. 82 de la Ley 26.912: empieza en la página 42 y sigue en la 43. Como el sistema carga y corta cada página por separado, los incisos d) a f) quedaron en otro fragmento que no llegó al modelo, y la respuesta enumeró sólo tres de las seis obligaciones. **Qué cambiaría:** unir las páginas de cada ley antes de cortar (guardando la página donde empieza cada fragmento) o cortar por artículo; también probaría subir `k` de 3 a 5.

---

## 7. Decisiones y límites

| Decisión | Qué elegí | Por qué |
|---|---|---|
| Limpieza previa | Quitar con expresiones regulares los pies de página, URL, enlaces `(/normativa/...)`, íconos y rótulos de la web | Al guardar desde el navegador, cada página trae texto que no es parte de la ley. Sin limpieza, ese texto entra a los fragmentos y al contexto. Se quitaron unos 25.000 caracteres y se comprobó que no quedaran restos. Los metadatos `source` y `page` no se modifican |
| Tamaño y solapamiento | `chunk_size = 1100`, `chunk_overlap = 180` (en caracteres) | Comparé 500/80 contra 1100/180 con la pregunta de clubes de barrio: con 500 el art. 2 de la Ley 27.098 queda partido; con 1100 entra entero (sección 5 del cuaderno). Los artículos de ley son largos y conviene que cada fragmento conserve una unidad de sentido |
| `k` | 3 fragmentos | Alcanzó para las preguntas de artículos cortos; con artículos largos faltaron partes (ver limitación) |
| Modelo de embeddings | `bge-m3` para indexar y para consultar | Es el de la ruta del curso y trabaja bien en español. Se usa el mismo modelo en el índice y en la pregunta |
| Indexación por lotes | Se agregan los fragmentos de a 32 | Con un único pedido de más de 600 fragmentos, el proceso de Ollama se cortaba (error de conexión con el *runner*). Por lotes funciona y muestra el avance |
| Prompt | Responder sólo con el contexto, citar [1], [2] o [3] y, si falta el dato, responder exactamente «No encuentro esa información en los documentos.» | Se considera evidencia sólo lo que está en los fragmentos. El prompt pide abstención, pero no la garantiza |

**Límites**
- **Páginas:** cada página se fragmenta por separado, así que un artículo que cruza de página queda dividido.
- **Primera página:** el texto de la primera página de cada PDF mezcla datos de la web (sanción, Boletín Oficial) con el inicio de la ley, porque así lo extrae el cargador.
- **Desbalance:** la Ley 26.912 aporta la mayor parte de los fragmentos.
- **Citas:** la lista de fuentes muestra **qué fragmentos se enviaron al modelo**, no que el modelo los haya interpretado bien. En las pruebas el modelo citó fragmentos que no usó. Hay que verificar cada respuesta contra el fragmento y el PDF.
- **Vigencia:** las leyes pueden modificarse después de la fecha de descarga (7/11/2026). La Ley 27.098 se incluyó en su texto original.
- **No es asesoramiento legal.**

---

## 8. Uso de IA

Usé **Claude (Anthropic), modelo Claude Opus 5.5**, entre el 5 y el 8 de noviembre de 2026, como asistente durante todo el trabajo:

| Momento | Para qué lo usé |
|---|---|
| Planificación (5/11) | Entender la consigna y armar un plan de trabajo |
| Corpus (7/11) | Buscar leyes públicas posibles para el corpus, revisar que los PDF tuvieran texto extraíble y redactar la tabla de procedencia |
| Implementación (7/11) | Adaptar el cuaderno `07_laboratorio_rag_espanol.ipynb` del curso a tres documentos, escribir las reglas de limpieza del texto y adaptar `app.py` |
| Errores de instalación y ejecución (8/11) | Resolver el problema de OneDrive con `uv`, la política de ejecución de PowerShell y la caída de Ollama al indexar (solución por lotes) |
| Análisis de pruebas (8/11) | Revisar las respuestas contra los PDF y redactar el diagnóstico y este README |

**Lo que verifiqué personalmente:**
- Descargué los tres documentos de argentina.gob.ar y comprobé que fueran la versión indicada.
- Instalé el entorno y ejecuté el cuaderno completo y la aplicación en mi computadora con Ollama.
- Hice las cinco preguntas de prueba en la interfaz y comparé los fragmentos recuperados con las páginas de los PDF.
- Revisé que la salida de la comparación de cortes coincidiera con la decisión tomada (con 500 caracteres el art. 2 de la Ley 27.098 queda partido; con 1100, entero).
- Revisé el diagnóstico y la limitación con los resultados que obtuve.

