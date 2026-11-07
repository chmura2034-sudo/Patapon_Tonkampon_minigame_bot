#!/usr/bin/env python3
"""
Bot do minigierki z kowadłem w Patapon 2 (poziomy 1-3).

Reguły są te same co w części 1: bot trzyma się rytmu ramki, a nie samego ruchu miecha.
Na podstawie nagrań wszystkich poziomów (bez klikania):
  * tempo zmienia się w rundach: 75 -> 90 -> 105 -> 75 BPM, okres ramki 800/667/571 ms,
  * naciśnięcie miecha widać ok. 80 ms po rozbłysku, niezależnie od tempa,
  * podwójne naciśnięcie: drugie wypada pół okresu ramki po pierwszym.
Po wykryciu miecha bot wciska klawisz w następnym punkcie rytmu z opóźnieniem dobranym
w części 1 i skalowanym z tempem. Pomija naciśnięcia poza rytmem i moment bez sztabki.

Poziom określa liczbę uderzeń: bot pokazuje postęp i po ostatnim sam się wstrzymuje, żeby
nie reagować na ruchy miecha po końcu gry. Zły poziom można zmienić w trakcie pracy.

Bot startuje WSTRZYMANY: klawisz wysyła dopiero po F8, do okna, które ma fokus.

Sterowanie:
  F8        start / pauza
  F4        od nowa: zeruje licznik, kasuje zaplanowane naciśnięcia i włącza bota
  F6 / F7   przesunięcie -10 ms / +10 ms (działa w trakcie gry)
  F9        koniec (Ctrl+C w terminalu też kończy)

Te same polecenia działają ze standardowego wejścia (przelacz, restart, minus, plus,
koniec, "poziom N"),
z którego korzysta panel. Linie "[stan] ..." i "[pole] Nazwa: wartość" odczytuje panel.

Przykłady:
  .venv/bin/python panel.py                                 # panel z listą botów
  .venv/bin/python boty/kowadlo2/bot.py                      # bez panelu
  .venv/bin/python boty/kowadlo2/bot.py --przesuniecie 0.14  # 10 ms wcześniej
  .venv/bin/python boty/kowadlo2/bot.py --debug              # też pominięte naciśnięcia

Każda próba trafia do badania/bot_<data_godzina>.csv.
"""
import argparse
import csv
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import mss
import numpy as np
from pynput import keyboard

from badanie import (DODATKOWE, MAX_NIEBIESKI, MIECH, MIN_ODSTEP_NACISNIEC, OKNO_SPADKU, PAS,
                     RAMKI, REF_W, SCHEMATY, SPADEK_MIECHA, jasnosc, jasnosc_krawedzi,
                     obszar_16_9, piksele, region)

OKRES_RAMKI = 0.80005  # [s] tempo podstawowe; siatka dopasowuje się dalej sama
PO_SIATCE = 0.080  # [s] naciśnięcie miecha widać tyle po rozbłysku, przy każdym tempie
SZCZYT_RAMKI = 0.054  # [s] od zapalenia ramki do szczytu jasności
PRZESUNIECIE = 0.150  # [s] wartość z części 1 przy OKRES_RAMKI; skalowana z tempem
TOLERANCJA = 0.04  # [s] dalej od oczekiwanego momentu uznajemy wykrycie za fałszywe
RAMKA_JASNA, RAMKA_CIEMNA = 150, 60
SZTABKA_JEST = 170  # średnia jasność regionu; bez sztabki ok. 120-145, ze sztabką ok. 232
MIN_ROZBLYSKOW = 4  # po zmianie tempa rytm wraca po ok. 2 s, przed startem rundy
ZNANE_OKRESY = (OKRES_RAMKI, 0.6667, 0.5714)  # [s] 75, 90 i 105 BPM; kilka rozbłysków daje błąd do 2%
TOLERANCJA_OKRESU = 0.03  # względna: dalej od znanych okresów ufamy samemu pomiarowi
OKRES_MIN, OKRES_MAX = 0.5, 1.1  # [s] dłuższy odstęp to pewnie pominięty rozbłysk
MAKS_CISZA = 3.0  # [s] bez rozbłysku siatka przestaje być wiarygodna


def liczba_uderzen(poziom):
    """Suma schematu: 1 to jedno uderzenie, 2 to dwa, a 0 (przerwa) nic nie dodaje."""
    return sum(map(sum, SCHEMATY[poziom]))


def znany_okres(T):
    """Dociąga zmierzony okres do znanego tempa gry, jeśli jest blisko."""
    najblizszy = min(ZNANE_OKRESY, key=lambda z: abs(T - z))
    return najblizszy if abs(T - najblizszy) <= TOLERANCJA_OKRESU * najblizszy else T


class Siatka:
    """Dopasowuje t = t0 + n * T do ostatnich rozbłysków ramki."""

    def __init__(self, okres, okno=16):
        self.T = okres
        self.t0 = None
        self.czasy = deque(maxlen=okno)

    def rozblysk(self, t):
        if len(self.czasy) >= 3:
            n = round((t - self.t0) / self.T)
            if abs(t - (self.t0 + n * self.T)) > 0.05:  # zmiana tempa: mierz od nowa
                self.czasy.clear()
        if len(self.czasy) == 1 and OKRES_MIN < t - self.czasy[-1] < OKRES_MAX:
            self.T = znany_okres(t - self.czasy[-1])
        self.czasy.append(t)
        self.t0 = t
        if len(self.czasy) >= 3:
            czasy = np.array(self.czasy)
            n = np.round((czasy - t) / self.T)
            if len(set(n)) == len(n):
                self.T = znany_okres(float(np.polyfit(n, czasy, 1)[0]))
                self.t0 = float((czasy - n * self.T).mean())

    def gotowa(self, teraz):
        return len(self.czasy) >= MIN_ROZBLYSKOW and teraz - self.czasy[-1] < MAKS_CISZA

    def punkt_blisko(self, t):
        krok = self.T / 2
        return self.t0 + round((t - self.t0) / krok) * krok


def parse_args():
    p = argparse.ArgumentParser(description="Bot wciskający klawisz w rytmie ramki")
    p.add_argument("--przesuniecie", type=float, default=PRZESUNIECIE,
                   help="[s] względem szczytu ramki w następnym punkcie siatki")
    p.add_argument("--poziom", type=int, choices=sorted(SCHEMATY), default=1,
                   help="numer poziomu (wyznacza liczbę uderzeń)")
    p.add_argument("--klawisz", default="d", help="klawisz do wciśnięcia")
    p.add_argument("--hold", type=float, default=0.05, help="jak długo trzymać klawisz [s]")
    p.add_argument("--monitor", type=int, default=1, help="monitor z grą wg mss (domyślnie 1)")
    p.add_argument("--okres", type=float, default=OKRES_RAMKI, help="początkowy okres ramki [s]")
    p.add_argument("--bez-sztabki", action="store_true",
                   help="naciskaj także, gdy nie widać sztabki na kowadle")
    p.add_argument("--debug", action="store_true", help="wypisuj pominięte naciśnięcia")
    return p.parse_args()


def main():
    args = parse_args()
    kb = keyboard.Controller()
    wstrzymany = threading.Event()
    wstrzymany.set()
    stop = threading.Event()
    restart = threading.Event()
    przesuniecie = [args.przesuniecie]
    uderzenia = [0]
    poziom = [args.poziom]
    przebieg = [0]  # rośnie przy restarcie; naciśnięcia z poprzedniego przebiegu przepadają

    def pokaz_postep():
        print(f"[pole] Uderzenia: {uderzenia[0]}/{liczba_uderzen(poziom[0])}")

    def pokaz_przesuniecie():
        print(f"[pole] Przesunięcie: {przesuniecie[0] * 1000:+.0f} ms")

    def przelacz():
        if wstrzymany.is_set():
            wstrzymany.clear()
            print("[stan] WŁĄCZONY: bot naciska")
        else:
            wstrzymany.set()
            print("[stan] WSTRZYMANY: bot nie naciska")

    def zmien_przesuniecie(zmiana):
        przesuniecie[0] += zmiana
        pokaz_przesuniecie()

    def ustaw_poziom(wartosc):
        if wartosc.isdigit() and int(wartosc) in SCHEMATY:
            poziom[0] = int(wartosc)
            print(f"[bot] poziom {poziom[0]}")
            print(f"[pole] Poziom: {poziom[0]}")
            pokaz_postep()

    polecenia = {
        "przelacz": przelacz,
        "restart": restart.set,
        "minus": lambda: zmien_przesuniecie(-0.01),
        "plus": lambda: zmien_przesuniecie(0.01),
        "koniec": stop.set,
    }

    def czytaj_polecenia():
        for linia in sys.stdin:
            slowa = linia.split()
            if len(slowa) == 2 and slowa[0] == "poziom":
                ustaw_poziom(slowa[1])
            elif len(slowa) == 1 and slowa[0] in polecenia:
                polecenia[slowa[0]]()

    threading.Thread(target=czytaj_polecenia, daemon=True).start()

    def on_press(k):
        if k == keyboard.Key.f8:
            przelacz()
        elif k == keyboard.Key.f4:
            restart.set()
        elif k == keyboard.Key.f6:
            polecenia["minus"]()
        elif k == keyboard.Key.f7:
            polecenia["plus"]()
        elif k == keyboard.Key.f9:
            stop.set()
            return False

    nasluch = keyboard.Listener(on_press=on_press)
    nasluch.start()

    katalog = Path(__file__).resolve().parent / "badania"
    katalog.mkdir(exist_ok=True)
    plik_logu = katalog / f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log = open(plik_logu, "w", newline="", buffering=1)  # zapis na bieżąco
    zapis = csv.writer(log)
    zapis.writerow(["t_miech", "cel", "t_wcisniecia", "przesuniecie_ms", "wynik"])

    def wcisnij(t_miech, cel, przes, numer_przebiegu):
        while (zostalo := cel - time.perf_counter()) > 0:
            time.sleep(zostalo - 0.002 if zostalo > 0.003 else 0)
        if numer_przebiegu != przebieg[0]:
            return
        t = time.perf_counter()
        kb.press(args.klawisz)
        time.sleep(args.hold)
        kb.release(args.klawisz)
        zapis.writerow([f"{t_miech:.4f}", f"{cel:.4f}", f"{t:.4f}", f"{przes * 1000:.0f}", "ok"])
        uderzenia[0] += 1
        pokaz_postep()
        print(f"[bot] '{args.klawisz}' {(t - t_miech) * 1000:.0f} ms po miechu, "
              f"błąd planu {(t - cel) * 1000:+.1f} ms")
        if uderzenia[0] >= liczba_uderzen(poziom[0]) and not wstrzymany.is_set():
            wstrzymany.set()
            print(f"[stan] WSTRZYMANY: poziom {poziom[0]} ukończony, F4 = od nowa")

    def pomin(t_miech, powod):
        zapis.writerow([f"{t_miech:.4f}", "", "", "", powod])
        if args.debug:
            print(f"[bot] pominięto: {powod}")

    with mss.MSS() as sct:
        monitory = sct.monitors[1:]
        if not 1 <= args.monitor <= len(monitory):
            print(f"[bot] nie ma monitora {args.monitor}; dostępne: 1-{len(monitory)}")
            nasluch.stop()
            return
        obszar = obszar_16_9(monitory[args.monitor - 1])
        skala = obszar["width"] / REF_W

        r_pas = region(obszar, PAS)
        r_sztabka = region(obszar, DODATKOWE["sztabka"])

        def wyciecie(rect):
            r = region(obszar, rect)
            x, y = r["left"] - r_pas["left"], r["top"] - r_pas["top"]
            return slice(y, y + r["height"]), slice(x, x + r["width"])

        w_miech = wyciecie(MIECH)
        w_ramka = wyciecie(RAMKI["ramka_dol"])

        print(f"[bot] monitor {args.monitor}: {obszar}")
        print(f"[bot] log: {plik_logu}")
        print("[bot] F8 = start/pauza, F4 = od nowa, F6/F7 = -/+10 ms, F9 = koniec")
        pokaz_przesuniecie()
        print(f"[pole] Poziom: {poziom[0]}")
        pokaz_postep()
        print("[stan] WSTRZYMANY: bot nie naciska")

        siatka = Siatka(args.okres)
        ramka_ciemna = False
        byla_gotowa = False
        zgloszony_okres = args.okres
        sztabka = False
        ostatnie = deque()  # (t, wysokość miecha) z okna OKNO_SPADKU
        ostatni_nacisk = 0.0
        ostatni_cel = 0.0
        numer = 0

        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                if restart.is_set():
                    restart.clear()
                    przebieg[0] += 1
                    uderzenia[0] = 0
                    ostatni_cel = 0.0
                    wstrzymany.clear()
                    zapis.writerow([f"{t0:.4f}", "", "", "", "restart"])
                    pokaz_postep()
                    print("[stan] WŁĄCZONY: od nowa (F4), bot naciska")
                pas = piksele(sct.grab(r_pas))
                t = (t0 + time.perf_counter()) / 2

                y, x = w_ramka
                jas_ramki = jasnosc_krawedzi(pas[y, x, :3].mean(axis=2))
                if jas_ramki < RAMKA_CIEMNA:
                    ramka_ciemna = True
                elif jas_ramki >= RAMKA_JASNA and ramka_ciemna:
                    ramka_ciemna = False
                    siatka.rozblysk(t)

                gotowa = siatka.gotowa(t)
                if gotowa != byla_gotowa:
                    byla_gotowa = gotowa
                    print(f"[bot] siatka {'złapana' if gotowa else 'utracona'}: "
                          f"okres ramki {siatka.T * 1000:.1f} ms")
                    print(f"[pole] Rytm: {siatka.T * 1000:.0f} ms" if gotowa else "[pole] Rytm: brak")
                    if gotowa:
                        zgloszony_okres = siatka.T
                elif gotowa and abs(siatka.T - zgloszony_okres) > 0.02:
                    zgloszony_okres = siatka.T
                    print(f"[bot] nowe tempo: okres ramki {siatka.T * 1000:.1f} ms")
                    print(f"[pole] Rytm: {siatka.T * 1000:.0f} ms")

                numer += 1
                if numer % 3 == 0:
                    sztabka = jasnosc(sct.grab(r_sztabka)).mean() >= SZTABKA_JEST

                y, x = w_miech
                miech = float((pas[y, x, 0] < MAX_NIEBIESKI).sum(axis=0).mean()) / skala
                ostatnie.append((t, miech))
                while t - ostatnie[0][0] > OKNO_SPADKU:
                    ostatnie.popleft()
                spadek = max(v for _, v in ostatnie) - miech

                if spadek >= SPADEK_MIECHA and t - ostatni_nacisk > MIN_ODSTEP_NACISNIEC:
                    ostatni_nacisk = t
                    if wstrzymany.is_set():
                        pass
                    elif not gotowa:
                        pomin(t, "brak siatki")
                    elif not sztabka and not args.bez_sztabki:
                        pomin(t, "brak sztabki")
                    else:
                        punkt = siatka.punkt_blisko(t - PO_SIATCE)
                        if abs(t - PO_SIATCE - punkt) > TOLERANCJA:
                            pomin(t, "poza siatką")
                        else:
                            opoznienie = SZCZYT_RAMKI + przesuniecie[0]
                            cel = punkt + siatka.T * (0.5 + opoznienie / OKRES_RAMKI)
                            if cel - ostatni_cel > 0.1:
                                ostatni_cel = cel
                                threading.Thread(target=wcisnij, daemon=True,
                                                 args=(t, cel, przesuniecie[0],
                                                       przebieg[0])).start()
                time.sleep(0.001)
        except KeyboardInterrupt:
            pass

    nasluch.stop()
    log.close()
    print("[bot] koniec")
    print("[stan] ZATRZYMANY")


if __name__ == "__main__":
    main()
