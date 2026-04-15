# Dokumentacja projektu NLP - vision RAG
## Dokumentacja dot. tworzenia korpusu do systemu RAG

1. Dokumentacja bibliotek musi zostać sciągnięta i następnie wyczyszczona do takiej postaci, aby można było ją wykorzystać w dwóch pipeline'ach:
    - Dense Retrieval: baza wektorowa ChromaDB i przekazanie embedingów do systemu RAG
    - Sparse Retrieval: wyszukanie najbardziej istotnych plików z punktu widzenia zapytania i przekazanie do systemu RAG

### Scraping

W pierwszym kroku wykonywany jest scraping polegający na ściągnięciu najnowszych wersji bibliotek `opencv`, `PCL` oraz `Open3D`. Wykonywane jest to za pomocą skryptów:
- `scrape_open3d.py`
- `scrape_opencv.py`
- `scrape_pcl.py`

Zapisuja one dokumentację w folderach `data/raw/<lib_name>`.
### Parsing

Po pobraniu dokumentacji następuje etap parsingu, czyli wyciągnięcia istotnych informacji z dokumentacji i odfiltrowania nieistotnych części (takich jak fragmenty kodu HTML), które mogłyby zaburzyć proces wyszukiwania informacji (Sparse) lub generacji generacji embedingów (Dense).

Do parsowania pobranych dokumentacji wykorzystywane są skrypty:
- `parse_open3d.py`
- `parse_opencv.py`
- `parse_pcl.py`

openCV oraz PCL opierają swoją dokumentację na generatorze Doxygen, natomiast Open3D wykorzystuje do dokumentacji bibliotekę Sphinx. W efekcie dla każdej biblioteki powstaje dokumentacja w postaci plików `.html`, ale ich struktura jest różna dla dokumentacji bazujących na Doxygenie i Sphinx.

Wpływa to na lekko odmienny sposób parsowania kodu w postaci plików `.html` do chunków danych zapisywanych w plikach `.json`. Niezależnie od biblioteki pojedynczy chunk ma następującą strukturę:


| Pole | Opis | Zastosowanie|
| :--- | :--- | :--- |
| `entity_name` | Nazwa funkcji lub klasy w C++ / Python | Sparse (BM25) - Heavy Boost |
| `signature` | The raw code signature (arguments, types). | Sparse (BM25) |
| `description` | Opis co robi dana funkcja | Sparse (BM25) |
| `template_parameters` | Argumenty szablonu C++ (głównie dla biblioteki PCL) | Sparse (BM25) |
| `parameters` | Argumenty funkcji i ich opisy | Sparse (BM25) |
| `returns` | Opis danych zwracanych przez funkcję | Sparse (BM25) |
| `dense_embedding_text` | Tekst składający się z wyżej wymienionych pól, aby dostarczyć kontekst do generacji embedingu | Dense (Vector DB) |
| `metadata` | Słownik składający się z nazwy biblioteki z której pochodzi fragment, nazwy pliku i hierarchi funkcji/klasy w bibliotece | Filtracja i kontekst |

### Vectorstore

Do stworzenia wektorowej bazy danych wykorzystano bibliotekę `Chroma` w połączeniu z frameworkiem `LangChain`. Kod dostępny jest w skrypcie `embedding.py`

Modele sprawdzone w procesie tworzenia embeddingów:

| Model | Liczba parametrów | Wielkość kontekstu | Wielkość modelu |
| ----- | ----------------  | -----------------  | --------------- |
| qwen3-embedding | 8B | 40K | 4.7 GB |

Jakość wyników zwracanych przez wektorową bazę danych można sprawdzić z wykorzystaniem skryptu `ask_vectorstore.py`.
## System RAG

Na ten moment zaimplementowano testowo system RAG wyłącznie z techniką Dense Retrieval. Kod dostępny w skrypcie `system.py`

Lista z przetestowanymi modelami LLM pracujacymi w systemie RAG:

| Model | Liczba parametrów | Wielkość kontekstu | Wielkość modelu |
| ----- | ----------------  | -----------------  | --------------- |
| gemma4:e4b | 8B | 128K | 9.6 GB |

> Note
> Zauważono, że przy zapytaniach dot. bibliotek wizji 3D system preferuje Open3D ponad PCL, dlatego dodano wyrażenie regularne, które filtruje odpowiedzi z wektorowej bazy danych jeżeli użytkownik w swoim zapytaniu wyszczególni konkretną bibliotekę.
>

