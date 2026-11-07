#!/usr/bin/env python3
"""
Badanie minigierki z kowadłem w Patapon 2 - TYLKO OBSERWACJA, poziom po poziomie.

Skrypt niczego nie naciska: nie wysyła żadnego wejścia do gry. Klawisz F6 jest tylko
nasłuchiwany. Nagrywa wyłącznie jeden wybrany monitor. Nie używa screenów wzorcowych:
scenę zapamiętuje w chwili F6 (naciśnij go, gdy minigierka jest już na ekranie).

Po nagraniu raport odpowiada na pytania:

  * czy regiony z części 1 (miech, ramka, sztabka) działają też tutaj,
  * jakie tempa występują i gdzie się zmieniają,
  * czy miech trzyma rytm ramki,
  * jaką sekwencję 1/2/przerwa pokazała gra i czy zgadza się ze schematem poziomu.

Użycie:
  .venv/bin/python boty/kowadlo2/badanie.py --poziom 1      # F6 = start, drugie F6 = stop
  .venv/bin/python boty/kowadlo2/badanie.py --poziom 2 --monitor 2
  .venv/bin/python boty/kowadlo2/badanie.py --analiza boty/kowadlo2/badania/poziom1_20261106_120000

Wyniki trafiają do badania/poziom<N>_<data_godzina>/:
  raport.txt     - wnioski
  probki.csv     - pomiary (setki próbek na sekundę)
  miniatury.npz  - miniatury całego ekranu (mapa tego, co pulsuje)
  klatki/*.jpg   - pełne zrzuty co sekundę i tuż po ruchu miecha
"""
import argparse
import csv
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import mss
import numpy as np
from PIL import Image
from pynput import keyboard

KATALOG = Path(__file__).resolve().parent
REF_W, REF_H = 1920, 1080  # współrzędne regionów pochodzą z góra.png / dół.png

# Regiony (x1, y1, x2, y2). Ramka ma 4 px grubości; krawędź dolna, lewa i prawa
# są mierzone na czarnym pasie, więc gwiazdy i tło nie zakłócają odczytu.
MIECH = (770, 795, 826, 885)
RAMKI = {
    "ramka_dol": (100, 1060, 640, 1064),
    "ramka_gora": (300, 20, 1600, 24),
    "ramka_lewa": (20, 900, 24, 1050),
    "ramka_prawa": (1896, 900, 1900, 1050),
}
DODATKOWE = {
    "sztabka": (860, 438, 1040, 456),  # rozgrzana sztabka na kowadle
    "piec": (900, 780, 1020, 835),  # palenisko w "ustach" kowadła
}
# Wspólny pas: miech, trzy krawędzie ramki i palenisko z jednego odczytu. Jeden duży
# odczyt jest równiejszy niż kilka małych i daje sygnały z tej samej chwili.
PAS = (20, 780, 1900, 1064)
W_PASIE = {
    "miech": MIECH,
    "ramka_dol": RAMKI["ramka_dol"],
    "ramka_lewa": RAMKI["ramka_lewa"],
    "ramka_prawa": RAMKI["ramka_prawa"],
    "piec": DODATKOWE["piec"],
}
POZA_PASEM = ("ramka_gora", "sztabka")  # odczytywane co ROTACJA próbek, na zmianę
ROTACJA = 3
KOLUMNY = ["t", "miech", *RAMKI, *DODATKOWE, "nowa_klatka"]

# Miech: liczymy po kanale niebieskim. Pomarańczowa poświata z pieca rozjaśnia czarny
# kolor, ale prawie nie dodaje niebieskiego, a jasne tło ma go dużo.
MAX_NIEBIESKI = 55
SPADEK_MIECHA = 20  # o tyle px miech musi spaść w OKNO_SPADKU, by liczyć naciśnięcie
OKNO_SPADKU = 0.12  # [s]; animacja: 58 -> 36 -> 15 px w dwóch krokach
MIN_ODSTEP_NACISNIEC = 0.25  # [s]
# Schematy poziomów: 1 = pojedyncze naciśnięcie, 2 = podwójne, 0 = przerwa. Długość
# przerwy nie jest tu zapisana, dlatego porównanie sklejamy ją do jednego znaku.
SCHEMATY = {
    1: ((1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1),  # odczytane z gry
        (1, 1, 1, 0, 1, 0, 1, 1),
        (1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1),
        (1, 1, 1, 1)),
    2: ((1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2),
        (1, 1, 2, 0, 1, 2, 2),
        (1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 0, 2, 2, 2, 1),
        (2, 1, 1, 1)),
    3: ((1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2),
        (2, 2, 1, 0, 2, 2, 1, 2),
        (1, 2, 1, 0, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1),
        (1, 1, 1, 2)),
}
CISZA_MIEDZY_RUNDAMI = 6.0  # [s] dłuższa przerwa w naciśnięciach to nowa runda
TOLERANCJA_ODCZYTU = 0.12  # część kroku elementu; dalej znak "?"
ZGODNOSC_TEMPA = 0.08  # względna zmiana okresu ramki, od której zaczyna się nowy odcinek
MIN_ROZBLYSKOW_ODCINKA = 3  # krótszy odcinek to drganie pomiaru
MIN_ZAKRES_MIECHA, MIN_ZAKRES_RAMKI = 20, 50  # mniejszy zakres: region nie trafia w cel
OKNO_SIATKI = 0.08  # [s] szerokość skupiska naciśnięć wokół punktu siatki
MIN_UDZIAL = 0.5  # co najmniej tyle naciśnięć musi trafić w skupisko
MIN_NAD_PRZYPADEK = 2.5  # i tyle razy więcej, niż wyszłoby przypadkiem
MINI_W, MINI_H = 96, 54
MINI_HZ = 15
MIN_ZMIANA = 8.0  # mniejsza zmiana sygnału to szum, a nie zdarzenie
PROG_SCENY = 25.0  # maks. średnia różnica miniatury od sceny zapamiętanej przy F6


# ---------------------------------------------------------------------------
# Obszar gry na monitorze
# ---------------------------------------------------------------------------

def obszar_16_9(widok):
    """Wycina z widoku obszar 16:9 (czarne pasy przy innych proporcjach)."""
    skala = min(widok["width"] / REF_W, widok["height"] / REF_H)
    w, h = round(REF_W * skala), round(REF_H * skala)
    return {
        "left": widok["left"] + (widok["width"] - w) // 2,
        "top": widok["top"] + (widok["height"] - h) // 2,
        "width": w,
        "height": h,
    }


def region(obszar, rect):
    skala = obszar["width"] / REF_W
    x1, y1, x2, y2 = rect
    return {
        "left": obszar["left"] + int(x1 * skala),
        "top": obszar["top"] + int(y1 * skala),
        "width": max(1, round((x2 - x1) * skala)),
        "height": max(1, round((y2 - y1) * skala)),
    }


def piksele(shot):
    """Zrzut mss -> tablica BGRA (h, w, 4)."""
    return np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)


def jasnosc(shot):
    """Zrzut mss -> tablica jasności (h, w). Tylko dla małych regionów."""
    return piksele(shot)[:, :, :3].mean(axis=2)


def miniatura(obraz):
    """Próbkuje piksele zamiast uśredniać cały obraz: 0,2 ms zamiast ponad 40 ms."""
    wiersze = np.linspace(0, obraz.shape[0] - 1, MINI_H).astype(int)
    kolumny = np.linspace(0, obraz.shape[1] - 1, MINI_W).astype(int)
    return obraz[np.ix_(wiersze, kolumny)][:, :, :3].mean(axis=2).astype(np.float32)


def roznica_od_sceny(mini, wzorce):
    return min(float(np.abs(mini - w).mean()) for w in wzorce)


# ---------------------------------------------------------------------------
# Nagrywanie
# ---------------------------------------------------------------------------

class Klatkarz(threading.Thread):
    """Wątek w tle: miniatury całego ekranu, zrzuty JPG, pilnowanie sceny."""

    def __init__(self, obszar, wzorce, katalog):
        super().__init__(daemon=True)
        self.obszar, self.wzorce, self.katalog = obszar, wzorce, katalog
        self.stop = threading.Event()
        self.zdarzenie = threading.Event()  # główna pętla prosi o zrzut
        self.zdarzen_zapisanych = 0
        self.czasy, self.mini = [], []
        self.ostatnio_widziana = None  # ostatnia chwila, gdy widać było scenę

    def zapisz(self, shot, nazwa):
        Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX").save(
            self.katalog / "klatki" / nazwa, quality=88
        )

    def run(self):
        (self.katalog / "klatki").mkdir(parents=True, exist_ok=True)
        ostatni_zrzut = 0.0
        with mss.MSS() as sct:
            while not self.stop.is_set():
                t0 = time.perf_counter()
                shot = sct.grab(self.obszar)
                t = (t0 + time.perf_counter()) / 2
                mini = miniatura(piksele(shot))
                self.czasy.append(t)
                self.mini.append(mini)
                if roznica_od_sceny(mini, self.wzorce) < PROG_SCENY:
                    self.ostatnio_widziana = t
                if self.zdarzenie.is_set():
                    self.zdarzenie.clear()
                    if self.zdarzen_zapisanych < 24:
                        self.zdarzen_zapisanych += 1
                        self.zapisz(shot, f"miech_{self.zdarzen_zapisanych:02d}.jpg")
                elif t - ostatni_zrzut > 1:
                    ostatni_zrzut = t
                    self.zapisz(shot, f"t_{len(self.mini):05d}.jpg")
                time.sleep(max(0.0, 1 / MINI_HZ - (time.perf_counter() - t0)))


def nagrywaj(obszar, wzorce, katalog, args, klawisz):
    skala = obszar["width"] / REF_W
    r_pas = region(obszar, PAS)
    wyciecia = {}
    for nazwa, rect in W_PASIE.items():
        r = region(obszar, rect)
        x, y = r["left"] - r_pas["left"], r["top"] - r_pas["top"]
        wyciecia[nazwa] = (slice(y, y + r["height"]), slice(x, x + r["width"]))
    r_poza = {n: region(obszar, {**RAMKI, **DODATKOWE}[n]) for n in POZA_PASEM}
    wartosci = dict.fromkeys(KOLUMNY[1:-1], 0.0)
    numer_probki = 0

    klatkarz = Klatkarz(obszar, wzorce, katalog)
    klatkarz.start()

    wiersze = []
    poprzedni_odcisk = None
    poprzedni_miech = None
    start = time.perf_counter()
    powod = "koniec zadanego czasu"
    print(f"[badanie] NAGRYWAM (limit {args.czas:.0f} s). F6 albo Ctrl+C kończy.")
    print(f"[stan] WŁĄCZONY: nagrywa poziom {args.poziom}")

    try:
        with mss.MSS() as sct:
            while True:
                t0 = time.perf_counter()
                if t0 - start >= args.czas:
                    break
                if klawisz.stop.is_set():
                    powod = "zatrzymane klawiszem F6"
                    break
                widziana = klatkarz.ostatnio_widziana
                if widziana is not None and t0 - widziana > args.po_utracie:
                    powod = f"scena zniknęła na ponad {args.po_utracie:.0f} s"
                    break

                pas = piksele(sct.grab(r_pas))
                t = (t0 + time.perf_counter()) / 2

                y, x = wyciecia["miech"]
                j_miech = pas[y, x, 0]  # BGRA, więc kanał 0 to niebieski
                miech = float((j_miech < MAX_NIEBIESKI).sum(axis=0).mean()) / skala
                wartosci["miech"] = miech
                for nazwa in ("ramka_dol", "ramka_lewa", "ramka_prawa"):
                    y, x = wyciecia[nazwa]
                    wartosci[nazwa] = float(jasnosc_krawedzi(pas[y, x, :3].mean(axis=2)))
                y, x = wyciecia["piec"]
                wartosci["piec"] = float(pas[y, x, :3].mean())

                numer_probki += 1
                if numer_probki % ROTACJA == 0:
                    nazwa = POZA_PASEM[(numer_probki // ROTACJA) % len(POZA_PASEM)]
                    jas = jasnosc(sct.grab(r_poza[nazwa]))
                    wartosci[nazwa] = float(jasnosc_krawedzi(jas) if nazwa in RAMKI
                                            else jas.mean())

                odcisk = hash(j_miech.tobytes())
                nowa = int(odcisk != poprzedni_odcisk)
                poprzedni_odcisk = odcisk

                if poprzedni_miech is not None and abs(miech - poprzedni_miech) > 8:
                    klatkarz.zdarzenie.set()
                poprzedni_miech = miech

                wiersze.append((t, *(wartosci[k] for k in KOLUMNY[1:-1]), nowa))
                time.sleep(0.001)
    except KeyboardInterrupt:
        powod = "przerwane przez Ctrl+C"

    klatkarz.stop.set()
    klatkarz.join(timeout=3)
    print(f"[badanie] stop: {powod}; próbek: {len(wiersze)}")

    # Po zniknięciu sceny (np. ekran wyników) dane psułyby progi analizy.
    koniec = klatkarz.ostatnio_widziana
    if koniec is not None:
        wiersze = [w for w in wiersze if w[0] <= koniec]
        zostaje = [i for i, t in enumerate(klatkarz.czasy) if t <= koniec]
        klatkarz.czasy = [klatkarz.czasy[i] for i in zostaje]
        klatkarz.mini = [klatkarz.mini[i] for i in zostaje]

    t0 = wiersze[0][0] if wiersze else 0.0
    with open(katalog / "probki.csv", "w", newline="") as f:
        zapis = csv.writer(f)
        zapis.writerow(KOLUMNY)
        for w in wiersze:
            zapis.writerow([f"{w[0] - t0:.5f}", *(f"{v:.2f}" for v in w[1:-1]), w[-1]])
    np.savez_compressed(
        katalog / "miniatury.npz",
        t=np.array(klatkarz.czasy, np.float64) - t0,
        mini=np.array(klatkarz.mini, np.float32),
    )
    (katalog / "info.txt").write_text(
        f"poziom={args.poziom}\nobszar={obszar}\npowod_konca={powod}\n")


def jasnosc_krawedzi(jas):
    """Jasność ramki: mediana wzdłuż krawędzi (ignoruje gwiazdy), najjaśniejszy pas."""
    wzdluz = 1 if jas.shape[1] >= jas.shape[0] else 0
    return np.median(jas, axis=wzdluz).max()


# ---------------------------------------------------------------------------
# Analiza
# ---------------------------------------------------------------------------

def przejscia(t, x):
    """Przerzutnik Schmitta. Zwraca (czasy wzrostu, czasy spadku, próg dół, próg góra)."""
    lo5, hi95 = np.percentile(x, [5, 95])
    if hi95 - lo5 < MIN_ZMIANA:
        return np.array([]), np.array([]), lo5, hi95
    dol, gora = lo5 + 0.35 * (hi95 - lo5), lo5 + 0.65 * (hi95 - lo5)
    wzrosty, spadki, stan = [], [], None
    for ti, xi in zip(t, x):
        if xi >= gora and stan != 1:
            if stan == 0:
                wzrosty.append(ti)
            stan = 1
        elif xi <= dol and stan != 0:
            if stan == 1:
                spadki.append(ti)
            stan = 0
    return np.array(wzrosty), np.array(spadki), dol, gora


def nacisniecia(t, miech):
    """Czasy szybkich spadków miecha. Powrót jest powolny, więc go nie łapiemy."""
    poczatki = np.searchsorted(t, t - OKNO_SPADKU)
    wynik = []
    for i, p in enumerate(poczatki):
        if (miech[p: i + 1].max() - miech[i] >= SPADEK_MIECHA
                and (not wynik or t[i] - wynik[-1] > MIN_ODSTEP_NACISNIEC)):
            wynik.append(t[i])
    return np.array(wynik)


def skupisko(fazy, T, okno):
    """Najgęstsze okno fazy na okręgu T. Zwraca (liczba w oknie, mediana fazy)."""
    f = np.sort(fazy)
    f2 = np.concatenate([f, f + T])
    ile = np.searchsorted(f2, f + okno, side="right") - np.arange(len(f))
    i = int(np.argmax(ile))
    return int(ile[i]), float(np.median(f2[i: i + ile[i]])) % T


def odcinki_tempa(rozblyski):
    """Dzieli rozbłyski na odcinki stałego tempa. Zwraca [(start, koniec, okres, liczba)]."""
    if len(rozblyski) < 2:
        return []
    odstepy = np.diff(rozblyski)
    odcinki, poczatek = [], 0
    for i in range(1, len(odstepy) + 1):
        if i == len(odstepy) or abs(odstepy[i] / np.median(odstepy[poczatek: i]) - 1) > ZGODNOSC_TEMPA:
            odcinki.append((float(rozblyski[poczatek]), float(rozblyski[i]),
                            float(np.median(odstepy[poczatek: i])), i - poczatek + 1))
            poczatek = i
    return [o for o in odcinki if o[3] >= MIN_ROZBLYSKOW_ODCINKA]


def fazy_lokalne(naciski, rozblyski):
    """Faza 0-1 każdego naciśnięcia w jego własnym okresie ramki (działa przy zmianie tempa)."""
    fazy, okresy = [], []
    if len(rozblyski) < 2:
        return np.array(fazy), np.array(okresy)
    typowy = float(np.median(np.diff(rozblyski)))
    for t in naciski:
        i = int(np.searchsorted(rozblyski, t, side="right")) - 1
        if 0 <= i < len(rozblyski) - 1:
            okres = rozblyski[i + 1] - rozblyski[i]
            if okres < 1.5 * typowy:
                fazy.append((t - rozblyski[i]) / okres)
                okresy.append(okres)
    return np.array(fazy), np.array(okresy)


def podzial_na_rundy(naciski):
    if not len(naciski):
        return []
    granice = np.flatnonzero(np.diff(naciski) > CISZA_MIEDZY_RUNDAMI) + 1
    return np.split(naciski, granice)


def odczytaj_runde(naciski, rozblyski):
    """Zwraca (sekwencja, krok elementu). Znaki: 1, 2, 0 = puste miejsce, ? = nie pasuje."""
    w_rundzie = rozblyski[(rozblyski >= naciski[0] - 2) & (rozblyski <= naciski[-1] + 2)]
    # Dopasowanie do wszystkich rozbłysków: błąd mediany narastałby przez całą rundę.
    dopasowanie = siatka(w_rundzie)
    okres = dopasowanie[0] if dopasowanie else 0.8
    krok = 2 * okres
    miejsca = {}
    for t in naciski:
        polozenie = (t - naciski[0]) / krok
        n = int(round(polozenie))
        miejsca.setdefault(n, []).append(polozenie - n)
    # Drugie uderzenie podwójnego wypada w 1/4 kroku, więc round() zostawia je przy pierwszym.
    sekwencja = []
    for n in range(max(miejsca) + 1):
        odchylenia = miejsca.get(n)
        if odchylenia is None:
            sekwencja.append("0")
        elif len(odchylenia) == 1 and abs(odchylenia[0]) <= TOLERANCJA_ODCZYTU:
            sekwencja.append("1")
        elif (len(odchylenia) == 2 and abs(odchylenia[0]) <= TOLERANCJA_ODCZYTU
              and abs(odchylenia[1] - 0.25) <= TOLERANCJA_ODCZYTU):
            sekwencja.append("2")
        else:
            sekwencja.append("?")
    return sekwencja, krok


def bez_dlugosci_przerw(elementy):
    wynik = []
    for e in elementy:
        if e != "0" or (wynik and wynik[-1] != "0"):
            wynik.append(e)
    return wynik


def porownaj_ze_schematem(naciski, rozblyski, poziom, rap):
    rap(f"\n--- ZGODNOŚĆ ZE SCHEMATEM POZIOMU {poziom} ---")
    schemat = [[str(e) for e in runda] for runda in SCHEMATY[poziom]]
    rundy = podzial_na_rundy(naciski)
    rap(f"Rund w nagraniu: {len(rundy)}, w schemacie: {len(schemat)}")
    zgodne = 0
    for i, runda in enumerate(rundy):
        odczyt, krok = odczytaj_runde(runda, rozblyski)
        rap(f"Runda {i + 1}: od {runda[0]:.2f} s, {len(runda)} naciśnięć, "
            f"krok {krok * 1000:.0f} ms, długość {len(odczyt)} miejsc")
        rap("  odczyt : " + " ".join("_" if e == "0" else e for e in odczyt))
        if i < len(schemat):
            rap("  schemat: " + " ".join("_" if e == "0" else e for e in schemat[i]))
            if bez_dlugosci_przerw(odczyt) == schemat[i]:
                zgodne += 1
                rap("  ZGODNA")
            else:
                rap("  RÓŻNICA")
        else:
            rap("  poza schematem")
    rap(f"Zgodne rundy: {zgodne} z {len(schemat)}")


def siatka(zdarzenia):
    """Dopasowuje stały okres: t_k = t0 + n_k * T. Zwraca (T, t0, odchylenie, n)."""
    if len(zdarzenia) < 3:
        return None
    T = float(np.median(np.diff(zdarzenia)))
    n = np.zeros(len(zdarzenia))
    t0 = float(zdarzenia[0])
    for _ in range(4):
        n = np.round((zdarzenia - t0) / T)
        if len(set(n)) < 2:
            return None
        T, t0 = (float(v) for v in np.polyfit(n, zdarzenia, 1))
    reszty = zdarzenia - (t0 + n * T)
    return T, t0, float(reszty.std()), n


def opis_okresu(nazwa, zdarzenia, rap):
    if len(zdarzenia) < 3:
        rap(f"{nazwa}: za mało zdarzeń ({len(zdarzenia)}), by policzyć okres")
        return None
    dt = np.diff(zdarzenia) * 1000
    wartosci, ile = np.unique(np.round(dt / 10) * 10, return_counts=True)
    rozklad = ", ".join(f"{w:.0f} ms x{c}" for w, c in zip(wartosci, ile))
    wynik = siatka(zdarzenia)
    rap(f"{nazwa}: {len(zdarzenia)} zdarzeń, odstęp mediana {np.median(dt):.1f} ms, "
        f"min {dt.min():.1f}, maks {dt.max():.1f}")
    rap(f"  rozkład odstępów: {rozklad}")
    if wynik:
        T, _, rozrzut, n = wynik
        pominiete = int(n.max() - n.min() + 1 - len(zdarzenia))
        rap(f"  stały okres {T * 1000:.2f} ms = {60 / T:.1f} na minutę, "
            f"rozrzut względem siatki {rozrzut * 1000:.1f} ms, pominięte: {pominiete}")
    return wynik


def profil(t, x, zero, T, rap, etykieta, kosze=20):
    faza = np.mod(t - zero, T)
    k = np.minimum((faza / T * kosze).astype(int), kosze - 1)
    srednie = np.array([x[k == i].mean() if (k == i).any() else np.nan for i in range(kosze)])
    lo, hi = np.nanmin(srednie), np.nanmax(srednie)
    rap(f"  {etykieta}:")
    for i, v in enumerate(srednie):
        pasek = "#" * int(30 * (v - lo) / max(hi - lo, 1e-6)) if np.isfinite(v) else ""
        rap(f"    {i * T / kosze * 1000:7.1f} ms {v:7.1f} |{pasek}")


def analiza(katalog):
    katalog = Path(katalog)
    linie = []

    def rap(tekst=""):
        print(tekst)
        linie.append(tekst)

    with open(katalog / "probki.csv") as f:
        kolumny = next(csv.reader(f))
    dane = np.loadtxt(katalog / "probki.csv", delimiter=",", skiprows=1, ndmin=2)
    if len(dane) < 10:
        rap("Za mało próbek do analizy.")
        (katalog / "raport.txt").write_text("\n".join(linie) + "\n", encoding="utf-8")
        return
    k = {nazwa: dane[:, i] for i, nazwa in enumerate(kolumny)}
    t = k["t"]

    rap(f"=== RAPORT {katalog.name} ===")
    info = katalog / "info.txt"
    if info.exists():
        rap(info.read_text().strip())
    czas = t[-1] - t[0]
    nowe = int(k["nowa_klatka"].sum())
    rap(f"Czas {czas:.1f} s; {len(t) / czas:.0f} próbek/s; "
        f"zmian obrazu miecha {nowe / czas:.1f}/s")
    rap(f"Czas trwania jednego pomiaru: mediana {np.median(np.diff(t)) * 1000:.1f} ms, "
        f"najdłuższa przerwa {np.diff(t).max() * 1000:.1f} ms")

    # --- miech ---
    rap("\n--- MIECH (wysokość czarnego w px) ---")
    miech = k["miech"]
    rap(f"Zakres: p5={np.percentile(miech, 5):.1f}, p95={np.percentile(miech, 95):.1f} "
        f"(część 1: pełne wciśnięcie ok. 15, góra ok. 58)")
    zakres_miecha = np.percentile(miech, 95) - np.percentile(miech, 5)
    if zakres_miecha < MIN_ZAKRES_MIECHA:
        rap(f"UWAGA: zakres miecha tylko {zakres_miecha:.1f} px, region MIECH prawdopodobnie "
            "nie trafia w miech. Trzeba go ustawić według klatek z nagrania.")
    naciski = nacisniecia(t, miech)
    rap(f"Naciśnięcia: {len(naciski)}")
    if len(naciski):
        rap("Czasy naciśnięć [s]: " + " ".join(f"{v:.3f}" for v in naciski[:60]))

    # --- ramka ---
    rap("\n--- RAMKA (jasność krawędzi) ---")
    ramka = k["ramka_dol"]
    rap(f"Zakres dolnej krawędzi: p5={np.percentile(ramka, 5):.1f}, "
        f"p95={np.percentile(ramka, 95):.1f} (część 1: 0 i ok. 250)")
    for nazwa in RAMKI:
        if nazwa != "ramka_dol" and k[nazwa].std() > 0 and ramka.std() > 0:
            uwaga = " (mierzona rzadziej, niższa wartość jest tu normalna)" \
                if nazwa in POZA_PASEM else ""
            rap(f"  korelacja {nazwa} z dolną: "
                f"{np.corrcoef(ramka, k[nazwa])[0, 1]:+.3f}{uwaga}")
    zakres_ramki = np.percentile(ramka, 95) - np.percentile(ramka, 5)
    if zakres_ramki < MIN_ZAKRES_RAMKI:
        rap(f"UWAGA: zakres ramki tylko {zakres_ramki:.1f}, region ramki prawdopodobnie nie "
            "trafia w ramkę. Trzeba go ustawić według klatek z nagrania.")
    r_wzrost, r_spadek, _, _ = przejscia(t, ramka)
    opis_okresu("Rozbłyski ramki (narastanie)", r_wzrost, rap)
    odcinki = odcinki_tempa(r_wzrost)
    if odcinki:
        rap("Odcinki stałego tempa:")
        for start, koniec, okres, ile in odcinki:
            rap(f"  {start:7.2f}-{koniec:7.2f} s: okres ramki {okres * 1000:6.1f} ms "
                f"= {60 / okres:5.1f} BPM, rozbłysków {ile}")
    if len(r_wzrost) and len(r_spadek):
        jasna = [s - w for w in r_wzrost for s in r_spadek[r_spadek > w][:1]]
        if jasna:
            rap(f"Ramka zostaje jasna: mediana {np.median(jasna) * 1000:.0f} ms")

    # --- synchronizacja ---
    rap("\n--- CZY RAMKA POKRYWA SIĘ Z RYTMEM MIECHA? ---")
    werdykt = "BRAK DANYCH: za mało zdarzeń miecha lub ramki"
    fazy, okresy = fazy_lokalne(naciski, r_wzrost)
    if len(fazy) >= 4:
        T_r = float(np.median(okresy))
        wybrana = None
        for dzielnik in (1, 2, 4):
            okno = OKNO_SIATKI / T_r * dzielnik
            ile, przes = skupisko(np.mod(fazy * dzielnik, 1.0), 1.0, okno)
            udzial = ile / len(fazy)
            rap(f"Krok {T_r / dzielnik * 1000:6.1f} ms: {udzial * 100:3.0f}% naciśnięć w jednym "
                f"skupisku (przypadkiem {okno * 100:.0f}%)")
            if wybrana is None and udzial >= MIN_UDZIAL and udzial >= MIN_NAD_PRZYPADEK * okno:
                wybrana = dzielnik, przes
        if wybrana:
            dzielnik, przes = wybrana
            blad = (np.mod(fazy * dzielnik - przes + 0.5, 1.0) - 0.5) * okresy / dzielnik
            trafione = np.abs(blad) <= 0.04
            rap(f"Naciśnięcia w rytmie (+/-40 ms): {trafione.sum()} z {len(fazy)}, "
                f"rozrzut {blad[trafione].std() * 1000:.1f} ms")
            rap(f"Naciśnięcie widać {przes * T_r / dzielnik * 1000:.1f} ms po punkcie rytmu "
                "(część 1: 98 ms)")
            werdykt = (f"TAK: miech trzyma rytm ramki, krok 1/{dzielnik} okresu "
                       f"({trafione.sum()} z {len(fazy)} naciśnięć)")
        else:
            werdykt = "NIE: naciśnięcia nie trzymają rytmu ramki"
        odc = max(odcinki, key=lambda o: o[3])
        q = (t >= odc[0]) & (t <= odc[1])
        profil(t[q], ramka[q], odc[0], odc[2], rap, "Ramka w jednym okresie (0 = rozbłysk)")
    rap(f"\nWERDYKT: {werdykt}")

    poziom = next((int(l.split("=")[1]) for l in
                   (info.read_text().splitlines() if info.exists() else ())
                   if l.startswith("poziom=")), None)
    if poziom in SCHEMATY and len(naciski):
        porownaj_ze_schematem(naciski, r_wzrost, poziom, rap)

    # --- co jeszcze pulsuje ---
    plik_mini = katalog / "miniatury.npz"
    T_rytmu = float(max(odcinki, key=lambda o: o[3])[2]) if odcinki else None
    if plik_mini.exists() and T_rytmu:
        m = np.load(plik_mini)
        if len(m["t"]) > 30:
            rap(f"\n--- CO PULSUJE W RYTMIE {T_rytmu * 1000:.1f} ms (mapa ekranu 16x9) ---")
            mini = m["mini"] - m["mini"].mean(axis=0)
            nosna = np.exp(-2j * np.pi * m["t"] / T_rytmu)
            amp = 2 * np.abs(np.tensordot(nosna, mini, axes=(0, 0))) / len(nosna)
            h, w = amp.shape[0] // 9, amp.shape[1] // 16
            mapa = amp[: h * 9, : w * 16].reshape(9, h, 16, w).max(axis=(1, 3))
            for wiersz in mapa:
                rap("  " + " ".join(f"{v:4.0f}" for v in wiersz))
            calosc = m["mini"].mean(axis=(1, 2))
            rap(f"Średnia jasność ekranu: rozrzut {calosc.std():.2f} (skala 0-255)")

    # --- zmienne dodatkowe ---
    rap("\n--- DODATKOWE ---")
    for nazwa in DODATKOWE:
        wz, sp, _, _ = przejscia(t, k[nazwa])
        rap(f"{nazwa}: p5-p95 {np.percentile(k[nazwa], 5):.0f}-"
            f"{np.percentile(k[nazwa], 95):.0f}, "
            f"pojawienia {len(wz)}, zaniknięcia {len(sp)}")
        if len(wz):
            rap("  pojawienia [s]: " + " ".join(f"{v:.3f}" for v in wz[:24]))

    (katalog / "raport.txt").write_text("\n".join(linie) + "\n", encoding="utf-8")
    print(f"\n[badanie] raport: {katalog / 'raport.txt'}")


# ---------------------------------------------------------------------------

class KlawiszF6:
    """Pierwsze F6 zaczyna nagrywanie, drugie je kończy. Tylko nasłuch."""

    def __init__(self):
        self.start = threading.Event()
        self.stop = threading.Event()
        self.nasluch = keyboard.Listener(on_press=self.nacisniety)
        threading.Thread(target=self.czytaj_polecenia, daemon=True).start()

    def czytaj_polecenia(self):
        """Polecenie "koniec" z panelu: kończy nagrywanie albo czekanie na start."""
        for linia in sys.stdin:
            if linia.strip() == "koniec":
                self.stop.set()
                self.start.set()

    def nacisniety(self, k):
        if k == keyboard.Key.f6:
            (self.stop if self.start.is_set() else self.start).set()


def parse_args():
    p = argparse.ArgumentParser(description="Badanie minigierki bez wysyłania wejścia do gry")
    p.add_argument("--poziom", type=int, choices=sorted(SCHEMATY), default=1,
                   help="numer poziomu (wybiera schemat do porównania)")
    p.add_argument("--czas", type=float, default=120, help="maks. czas nagrania [s]")
    p.add_argument("--po-utracie", type=float, default=8,
                   help="koniec, gdy sceny nie ma przez tyle sekund")
    p.add_argument("--monitor", type=int, default=1,
                   help="monitor z grą wg mss (1 = pierwszy, domyślnie 1)")
    p.add_argument("--analiza", metavar="KATALOG", help="tylko przeanalizuj zapisane nagranie")
    return p.parse_args()


def main():
    args = parse_args()
    if args.analiza:
        analiza(args.analiza)
        return

    with mss.MSS() as sct:
        monitory = sct.monitors[1:]
        if not 1 <= args.monitor <= len(monitory):
            print(f"[badanie] nie ma monitora {args.monitor}; dostępne: 1-{len(monitory)}")
            return
        monitor = monitory[args.monitor - 1]
        print(f"[badanie] monitor {args.monitor}: {monitor}")
        print(f"[pole] Poziom: {args.poziom}")
        print("[badanie] wejdź do minigierki i naciśnij F6, aby nagrywać (Ctrl+C = wyjście)")
        print("[stan] WSTRZYMANY: czeka na F6")

        widok = obszar_16_9(monitor)
        klawisz = KlawiszF6()
        klawisz.nasluch.start()
        try:
            while not klawisz.start.wait(0.2):
                pass
        except KeyboardInterrupt:
            klawisz.stop.set()
        if klawisz.stop.is_set():
            print("[badanie] wyjście, nic nie nagrano.")
            klawisz.nasluch.stop()
            return

        wzorce = [miniatura(piksele(sct.grab(widok)))]  # scena z chwili F6

    katalog = KATALOG / "badania" / datetime.now().strftime(f"poziom{args.poziom}_%Y%m%d_%H%M%S")
    katalog.mkdir(parents=True, exist_ok=True)
    nagrywaj(widok, wzorce, katalog, args, klawisz)
    klawisz.nasluch.stop()
    print("[stan] WSTRZYMANY: liczy raport")
    analiza(katalog)
    print("[stan] ZATRZYMANY")


if __name__ == "__main__":
    main()
