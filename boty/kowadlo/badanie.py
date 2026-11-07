#!/usr/bin/env python3
"""
Badanie minigierki z kowadłem (Ton Kampon) - TYLKO OBSERWACJA.

Skrypt niczego nie naciska i nie klika: nie wysyła żadnego wejścia do gry.
Klawisz F6 jest tylko nasłuchiwany. Nagrywa wyłącznie jeden wybrany monitor.
Po nagraniu odpowiada na pytania:

  * jak działa miech (wysokość w czasie, odstępy między naciśnięciami, BPM),
  * jak mruga ramka (okres, narastanie, gaśnięcie, czy 4 krawędzie mrugają razem),
  * czy mruganie ramki pokrywa się z rytmem miecha i z jakim przesunięciem,
  * co jeszcze na ekranie pulsuje w tym samym rytmie.

Użycie:
  .venv/bin/python boty/kowadlo/badanie.py                  # F6 = start, F6 drugi raz = stop
  .venv/bin/python boty/kowadlo/badanie.py --monitor 2      # gra na drugim monitorze
  .venv/bin/python boty/kowadlo/badanie.py --czas 180       # dłuższy limit nagrania
  .venv/bin/python boty/kowadlo/badanie.py --analiza badania/20261106_120000   # ponowna analiza

Wyniki trafiają do badania/<data_godzina>/:
  raport.txt     - wnioski
  probki.csv     - pomiary (setki próbek na sekundę)
  miniatury.npz  - miniatury całego ekranu (mapa tego, co pulsuje)
  klatki/*.jpg   - pełne zrzuty, m.in. tuż po ruchu miecha
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
# Schemat minigierki: 1 = pojedyncze naciśnięcie miecha, 2 = podwójne (drugie o krok
# siatki później). Elementy idą co 2 okresy ramki, czyli co 1600 ms.
SCHEMAT = (
    (1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1),
    (2, 2, 1, 2, 2, 2, 2, 1),
    (1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2),
    (1, 1, 2, 1),
)
MIN_PRZERWA_RUND = 5.0  # [s] krótsze zniknięcie sztabki nie oznacza nowej rundy
TOLERANCJA_SCHEMATU = 0.1  # [s]
OKNO_SIATKI = 0.08  # [s] szerokość skupiska naciśnięć wokół punktu siatki
MIN_UDZIAL = 0.5  # co najmniej tyle naciśnięć musi trafić w skupisko
MIN_NAD_PRZYPADEK = 2.5  # i tyle razy więcej, niż wyszłoby przypadkiem
MINI_W, MINI_H = 96, 54
MINI_HZ = 15
MIN_ZMIANA = 8.0  # mniejsza zmiana sygnału to szum, a nie zdarzenie
PROG_SCENY = 15.0  # maks. średnia różnica miniatury od wzorca, by uznać scenę


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


def wzorce_sceny():
    wynik = []
    for nazwa in ("góra.png", "dół.png"):
        wynik.append(miniatura(np.asarray(Image.open(KATALOG / nazwa).convert("RGB"))))
    return wynik


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
                elif t - ostatni_zrzut > 3:
                    ostatni_zrzut = t
                    self.zapisz(shot, f"co3s_{len(self.mini):05d}.jpg")
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
    (katalog / "info.txt").write_text(f"obszar={obszar}\npowod_konca={powod}\n")


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


def skupisko(fazy, T):
    """Najgęstsze okno fazy na okręgu T. Zwraca (liczba w oknie, mediana fazy)."""
    f = np.sort(fazy)
    f2 = np.concatenate([f, f + T])
    ile = np.searchsorted(f2, f + OKNO_SIATKI, side="right") - np.arange(len(f))
    i = int(np.argmax(ile))
    return int(ile[i]), float(np.median(f2[i: i + ile[i]])) % T


def oczekiwane_nacisniecia(start, runda, T_ramki):
    czasy = []
    for i, element in enumerate(runda):
        czasy.append(start + i * 2 * T_ramki)
        if element == 2:
            czasy.append(start + i * 2 * T_ramki + T_ramki / 2)
    return np.array(czasy)


def porownaj_ze_schematem(naciski, starty, T_ramki, rap):
    """Dopasowuje rundy do schematu po numerze pierwszej widzianej rundy."""
    rap("\n--- ZGODNOŚĆ ZE SCHEMATEM ---")
    koniec_nagran = naciski[-1] + 1 if len(naciski) else 0
    najlepsze = None
    for pierwsza in range(len(SCHEMAT)):
        wiersze, razem = [], 0
        for i, start in enumerate(starty):
            numer = (pierwsza + i) % len(SCHEMAT)
            ocz = oczekiwane_nacisniecia(start, SCHEMAT[numer], T_ramki)
            ocz = ocz[ocz <= koniec_nagran]
            trafione = [t for t in ocz if len(naciski)
                        and np.abs(naciski - t).min() <= TOLERANCJA_SCHEMATU]
            koniec = start + len(SCHEMAT[numer]) * 2 * T_ramki
            w_rundzie = naciski[(naciski >= start - TOLERANCJA_SCHEMATU) & (naciski < koniec)]
            nadmiar = sum(np.abs(ocz - t).min() > TOLERANCJA_SCHEMATU for t in w_rundzie) \
                if len(ocz) else len(w_rundzie)
            brak = [f"{t:.1f}" for t in ocz if t not in trafione]
            razem += len(trafione)
            wiersze.append(f"Runda {numer + 1} od {start:.2f} s: trafione {len(trafione)} "
                           f"z {len(ocz)}, nadmiarowe {nadmiar}, brak w [s]: {brak}")
        if najlepsze is None or razem > najlepsze[0]:
            najlepsze = razem, wiersze
    for wiersz in najlepsze[1]:
        rap(wiersz)


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
        f"(pełne wciśnięcie ok. 15, góra ok. 58)")
    naciski = nacisniecia(t, miech)
    rap(f"Naciśnięcia: {len(naciski)}")
    if len(naciski):
        rap("Czasy naciśnięć [s]: " + " ".join(f"{v:.3f}" for v in naciski[:60]))

    # --- ramka ---
    rap("\n--- RAMKA (jasność krawędzi) ---")
    ramka = k["ramka_dol"]
    rap(f"Zakres dolnej krawędzi: p5={np.percentile(ramka, 5):.1f}, "
        f"p95={np.percentile(ramka, 95):.1f} (wg screenów: 74 i 191)")
    for nazwa in RAMKI:
        if nazwa != "ramka_dol" and k[nazwa].std() > 0 and ramka.std() > 0:
            uwaga = " (mierzona rzadziej, niższa wartość jest tu normalna)" \
                if nazwa in POZA_PASEM else ""
            rap(f"  korelacja {nazwa} z dolną: "
                f"{np.corrcoef(ramka, k[nazwa])[0, 1]:+.3f}{uwaga}")
    r_wzrost, r_spadek, _, _ = przejscia(t, ramka)
    siatka_r = opis_okresu("Rozbłyski ramki (narastanie)", r_wzrost, rap)
    opis_okresu("Gaśnięcia ramki", r_spadek, rap)
    if len(r_wzrost) and len(r_spadek):
        jasna = [s - w for w in r_wzrost for s in r_spadek[r_spadek > w][:1]]
        if jasna:
            rap(f"Ramka zostaje jasna: mediana {np.median(jasna) * 1000:.0f} ms")

    # --- synchronizacja ---
    rap("\n--- CZY RAMKA POKRYWA SIĘ Z RYTMEM MIECHA? ---")
    werdykt = "BRAK DANYCH: za mało zdarzeń miecha lub ramki"
    if len(naciski) >= 4 and siatka_r:
        T_r, t_r = siatka_r[0], siatka_r[1]
        # Miech nie musi mieć stałego okresu; ważne, czy trafia w siatkę rytmu ramki.
        wybrana = None
        for dzielnik in (1, 2, 4):
            T_s = T_r / dzielnik
            ile, przes = skupisko(np.mod(naciski - t_r, T_s), T_s)
            udzial, przypadek = ile / len(naciski), OKNO_SIATKI / T_s
            rap(f"Siatka {T_s * 1000:6.1f} ms: {udzial * 100:3.0f}% naciśnięć w jednym "
                f"skupisku (przypadkiem {przypadek * 100:.0f}%)")
            if (wybrana is None and udzial >= MIN_UDZIAL
                    and udzial >= MIN_NAD_PRZYPADEK * przypadek):
                wybrana = T_s, przes
        if wybrana:
            T_s, przes = wybrana
            blad = np.mod(naciski - t_r - przes + T_s / 2, T_s) - T_s / 2
            w_siatce = int((np.abs(blad) <= 0.04).sum())
            rap(f"Naciśnięcia w siatce (+/-40 ms): {w_siatce} z {len(naciski)}, "
                f"rozrzut trafionych {blad[np.abs(blad) <= 0.04].std() * 1000:.1f} ms")
            rap(f"Naciśnięcie widać {przes * 1000:.1f} ms po punkcie siatki "
                "wyznaczonym przez rozbłysk ramki")
            rap(f"Od naciśnięcia do następnego punktu siatki: {(T_s - przes) * 1000:.1f} ms, "
                f"do kolejnego: {(2 * T_s - przes) * 1000:.1f} ms")
            krotki = np.round(np.diff(naciski) / T_s).astype(int)
            wartosci, ile = np.unique(krotki, return_counts=True)
            rap("Odstępy między naciśnięciami w krokach siatki: "
                + ", ".join(f"{w} x{c}" for w, c in zip(wartosci, ile)))
            werdykt = (f"TAK: miech trzyma siatkę ramki, krok {T_s * 1000:.1f} ms "
                       f"({w_siatce} z {len(naciski)} naciśnięć)")
        else:
            werdykt = "NIE: naciśnięcia nie trzymają siatki rytmu ramki"
        profil(t, ramka, t_r, T_r, rap, "Ramka w jednym okresie (0 = rozbłysk)")
    rap(f"\nWERDYKT: {werdykt}")

    if siatka_r:
        pojawienia = przejscia(t, k["sztabka"])[0]
        starty = [v for i, v in enumerate(pojawienia)
                  if i == 0 or v - pojawienia[i - 1] > MIN_PRZERWA_RUND]
        if starty:
            porownaj_ze_schematem(naciski, starty, siatka_r[0], rap)

    # --- co jeszcze pulsuje ---
    plik_mini = katalog / "miniatury.npz"
    T_rytmu = siatka_r[0] if siatka_r else None
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
    p.add_argument("--czas", type=float, default=90, help="maks. czas nagrania [s]")
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
        print("[badanie] wejdź do minigierki i naciśnij F6, aby nagrywać (Ctrl+C = wyjście)")

        wzorce = wzorce_sceny()
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

        r = roznica_od_sceny(miniatura(piksele(sct.grab(widok))), wzorce)
        if r >= PROG_SCENY:
            print(f"[badanie] UWAGA: obraz nie przypomina minigierki (różnica {r:.1f}, "
                  f"próg {PROG_SCENY}). Sprawdź --monitor; nagrywam mimo to.")

    katalog = KATALOG / "badania" / datetime.now().strftime("%Y%m%d_%H%M%S")
    katalog.mkdir(parents=True, exist_ok=True)
    nagrywaj(widok, wzorce, katalog, args, klawisz)
    klawisz.nasluch.stop()
    analiza(katalog)


if __name__ == "__main__":
    main()
