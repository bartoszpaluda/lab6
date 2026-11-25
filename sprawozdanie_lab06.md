# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy  
**Zajęcia nr / temat:** Laboratorium 6 — Projekt generacyjny, część 1: przygotowanie danych dla Char-RNN  
**Autor** Bartosz Paluda
**Data:** 25.11.2026  
**Link do Githuba:** 

## 1. Cel ćwiczenia

Celem ćwiczenia jest przygotowanie oraz weryfikacja potoku przetwarzania danych tekstowych w języku polskim na potrzeby modelowania języka na poziomie znaków (przewidywanie kolejnego znaku przez sieć rekurencyjną Char-RNN). Zakres prac obejmuje pozyskanie i oczyszczenie korpusu z otwartego źródła, podział danych po całych dokumentach na zbiory treningowy, walidacyjny i testowy, budowę słownika znaków na zbiorze treningowym wraz z obsługą tokenów nieznanych (`<UNK>`), utworzenie obiektów `Dataset` i `DataLoader` w PyTorch oraz zbadanie wpływu długości okna kontekstu (`SEKW`) na liczbę par uczących.

## 2. Metoda i implementacja

### Charakterystyka korpusu i preprocessing

* **Źródło i warunki użycia:** Zbiór danych `wikimedia/wikipedia` (konfiguracja `20231101.pl`, pierwsze 250 artykułów) pobrany ze strumienia **Hugging Face Datasets**. Treści polskojęzycznej Wikipedii udostępniane są na otwartej licencji **CC BY-SA 3.0** (Creative Commons Uznanie autorstwa – Na tych samych warunkach), co zezwala na ich swobodne wykorzystanie, przetwarzanie i analizę w celach edukacyjnych.
* **Uzasadnienie wyboru i charakterystyka tekstu:** Artykuły encyklopedyczne obejmują zróżnicowaną tematykę (informatyka, fizyka, historia, biologia, medycyna) oraz formalny styl popularnonaukowy. Tekst zawiera pełny zestaw polskich znaków diakrytycznych, poprawną interpunkcję, skrótowce, liczby oraz nazwy własne, co stanowi wymagający i realistyczny materiał treningowy dla znakowego modelu generacyjnego.
* **Sposób przygotowania i oczyszczenia:** Odrzucono bardzo krótkie strony przekierowań (`len(d) <= 300` znaków), uzyskując **242 pełne artykuły** o łącznej długości **1 943 058 znaków**. Ujednolicono znaki końca linii do `\n`, zredukowano wielokrotne znaki nowej linii oraz nadmiarowe spacje/tabulatory, pozostawiając wielkość liter i znaki interpunkcyjne niezbędne do generowania poprawnej składni.
* **Fragment tekstu po oczyszczeniu:**
  > AWK – interpretowany język programowania, którego główną funkcją jest wyszukiwanie i przetwarzanie wzorców w plikach lub strumieniach danych. Jest także nazwą programu początkowo dostępnego dla systemów operacyjnych będących pochodnymi UNIX-a, obecnie także na inne platformy.
  > AWK jest językiem, który w znacznym stopniu wykorzystuje tablice asocjacyjne...

### Podział danych i ustawienia batchowania

* **Proporcje podziału:** Zastosowano podział **80% / 10% / 10%** na poziomie **całych dokumentów** (193 dokumenty treningowe, 24 walidacyjne, 25 testowych) przed utworzeniem okien czasowych. Przypisanie całych artykułów do odrębnych zbiorów oraz tworzenie okien wyłącznie wewnątrz granic poszczególnych dokumentów eliminuje ryzyko wycieku danych (*data leakage*). Różnice w procentowym udziale znaków (84,2% / 9,4% / 6,4%) wynikają ze zróżnicowanej długości pojedynczych artykułów encyklopedycznych.
* **Słownik i kodowanie:** Słownik zbudowano wyłącznie na zbiorze treningowym, przypisując tokenowi specjalnemu `<UNK>` indeks `0` w celu obsługi znaków niewystępujących w treningu.
* **Parametry `DataLoader`:** Krok przesuwania okna `stride = 1`, przesunięcie sekwencji celu o `+1` znak względem wejścia, bazowa długość okna `SEKW = 60`, rozmiar paczki `batch_size = 128`, tasowanie (`shuffle=True`) włączone wyłącznie dla zbioru treningowego.

## 3. Wyniki

### 1. Podział korpusu i liczebności (dla `SEKW = 60`, `batch_size = 128`)

| Zbiór | Liczba dokumentów | Liczba znaków | Udział znaków | Liczba różnych znaków | Liczba par (`SEKW=60`) | Liczba batchy (`batch=128`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trening** | 193 | 1 635 826 | 84.2% | 406 | 1 624 054 | 12 688 |
| **Walidacja** | 24 | 183 017 | 9.4% | 208 | 181 554 | 1 419 |
| **Test** | 25 | 124 215 | 6.4% | 162 | 122 691 | 959 |
| **Razem** | **242** | **1 943 058** | **100.0%** | **—** | **1 928 299** | **15 066** |

### 2. Kontrola poprawności danych i batchy

* **Rozmiar słownika:** **407** znaków (`406` unikalnych znaków ze zbioru treningowego + token `<UNK>` pod indeksem `0`).
* **Test kodowania i dekodowania:** `decode(encode(s)) == s` — **100% zgodności**:
  * Tekst źródłowy: `'AWK – interpretowany język programowania, którego główną fun'`
  * Pierwsze 15 ID: `[35, 57, 45, 2, 378, 2, 75, 80, 86, 71, 84, 82, 84, 71, 86]`
  * Tekst odkodowany: `'AWK – interpretowany język programowania, którego główną fun'`
* **Obsługa znaków spoza słownika (`OOV`):**
  * W walidacji wykryto **32** unikalne znaki spoza słownika treningowego (m.in. litery alfabetu gruzińskiego, ormiańskiego, arabskiego i hebrajskiego z obcojęzycznych wtrąceń encyklopedycznych: `{'ე', 'ვ', 'ա', 'ב', 'غ', ...}`).
  * W teście wykryto **6** unikalnych znaków spoza słownika treningowego (m.in. pismo klinowe oraz greckie symbole: `{'𐎧', 'ῶ', '𐏁', '𐏂', 'µ', 'Ἄ'}`).
  * Wszystkie znaki OOV zostały poprawnie zmapowane na indeks `0` (`<UNK>`) bez przerywania działania programu.
* **Weryfikacja tensorów i par wejście–cel (po 1 batchu z każdego zbioru):**
  * **Trening:** kształt `X: (128, 60)`, `Y: (128, 60)`, zakres ID: `[1, 387]`
    * `X[0]`: `'rzone zakończenie ciąży w wyniku interwencji zewnętrznej, np'`
    * `Y[0]`: `'zone zakończenie ciąży w wyniku interwencji zewnętrznej, np.'`
  * **Walidacja:** kształt `X: (128, 60)`, `Y: (128, 60)`, zakres ID: `[1, 378]`
    * `X[0]`: `'Amper – jednostka natężenia prądu elektrycznego (nazywane te'`
    * `Y[0]`: `'mper – jednostka natężenia prądu elektrycznego (nazywane też'`
  * **Test:** kształt `X: (128, 60)`, `Y: (128, 60)`, zakres ID: `[1, 378]`
    * `X[0]`: `'Aureus, nummus aureus (złoty pieniądz), denarius aureus – zł'`
    * `Y[0]`: `'ureus, nummus aureus (złoty pieniądz), denarius aureus – zło'`

### 3. Wpływ długości okna `SEKW` na liczbę par wejście–cel

| Długość okna `SEKW` | Liczba par (Trening) | Liczba par (Walidacja) | Liczba par (Test) |
| :---: | :---: | :---: | :---: |
| **30** | 1 629 844 | 182 274 | 123 441 |
| **60** | 1 624 054 | 181 554 | 122 691 |
| **120** | 1 612 474 | 180 114 | 121 191 |
| **200** | 1 597 034 | 178 194 | 119 191 |

<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/91494b2c-4905-473a-8ee0-6a07d397e635" />
<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/736bb693-d151-4625-bbc0-2de64b6c7af3" />



## 4. Analiza i interpretacja

1. **Rozkład częstości znaków w zbiorze treningowym:**
   * Wykres słupkowy potwierdza silną skośność rozkładu typową dla języka naturalnego. Zdecydowanie najczęściej występuje spacja (`' '`, ponad 215 tys. wystąpień, ok. 13,2% wszystkich znaków), rozdzielająca wyrazy. Kolejne miejsca zajmują samogłoski oraz najczęstsze spółgłoski języka polskiego: `a` (~117 tys.), `i` (~107 tys.), `e` (~101 tys.), `o` (~101 tys.), `n` (~79 tys.), `r` (~66 tys.) oraz `z` (~65 tys.). Tak duża dysproporcja sprawia, że prosty model bazowy przewidujący stale spację uzyskałby ok. 13% trafności bez analizy kontekstu.
2. **Wpływ parametru `SEKW` na liczbę par:**
   * Ponieważ okna generowane są oddzielnie wewnątrz każdego dokumentu z krokiem $1$, zwiększenie długości okna o $\Delta S$ zmniejsza liczbę przykładów w każdym dokumencie o $\Delta S$, co dla zbioru liczącego $D$ dokumentów daje łączny spadek liczby par równy $D \cdot \Delta S$ (np. przejście z `SEKW=30` na `SEKW=60` w 193 dokumentach treningowych redukuje liczbę par dokładnie o $193 \cdot 30 = 5790$ przykładów: $1\,629\,844 - 1\,624\,054 = 5790$). W skali całego korpusu jest to spadek niewielki (poniżej 2% przy zmianie z 30 na 200), natomiast dłuższe okno pozwala sieci rekurencyjnej uchwycić szerszy kontekst gramatyczny i dłuższe słowa kosztem wyższego czasu propagacji wstecznej przez czas (BPTT).
3. **Specyfika danych encyklopedycznych (znaki OOV):**
   * Duży rozmiar słownika treningowego (406 unikalnych znaków) wynika z obecności w artykułach Wikipedii zapisów etymologicznych w alfabetach obcych (cyrylica, greka, pismo arabskie, znaki matematyczne). Pojawienie się w walidacji i teście pojedynczych rzadkich glifów (np. znaków staroperskiego pisma klinowego `𐎧`) dowodzi, że zaimplementowany mechanizm fallbacku na token `<UNK>` jest konieczny do stabilnego działania `DataLoadera` podczas ewaluacji.

## 5. Wnioski

1. **Izolacja zbiorów na poziomie dokumentów:** Podział korpusu po całych artykułach przed tworzeniem okien skutecznie zapobiega przenikaniu kontekstu między treningiem a walidacją i testem, a jednocześnie gwarantuje, że żadne okno treningowe nie łączy dwóch niepowiązanych tematycznie tekstów.
2. **Odporność potoku na rzadkie znaki:** Zbudowanie słownika wyłącznie na części treningowej wraz z dedykowanym tokenem `<UNK>` zapewniło bezbłędne kodowanie wszystkich podzbiorów mimo wystąpienia 38 nieznanych znaków obcojęzycznych w walidacji i teście.
3. **Optymalny dobór `SEKW` i gotowość do Laboratorium 7:** Przygotowane tensory wejścia i celu o wymiarach `[128, 60]` (ponad 1,62 mln przykładów treningowych) są w pełni zweryfikowane i gotowe do bezpośredniego wykorzystania w treningu modelu generacyjnego Char-RNN na kolejnych zajęciach.
