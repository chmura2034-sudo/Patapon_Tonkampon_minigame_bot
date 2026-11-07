#!/usr/bin/env python3
"""
Panel do uruchamiania botów.

  .venv/bin/python panel.py

Panel sam znajduje boty: każdy plik boty/<folder>/*.json to jedna pozycja na liście.
Aby dodać bota, utwórz folder ze skryptem i plikiem JSON (wzór: boty/kowadlo/bot.json):

  nazwa, opis   tekst widoczny w panelu
  skrypt        plik .py uruchamiany z katalogu bota
  opcje         pola formularza: arg, etykieta, typ (float/int/str/bool/wybor), domyslnie;
                wybor ma też listę "wartosci" i opcjonalne "polecenie": zmiana w trakcie
                pracy wysyła do bota "<polecenie> <wartość>"
  polecenia     przyciski: etykieta i polecenie wysyłane do standardowego wejścia bota

Bot pisze do standardowego wyjścia. Wszystko trafia do dziennika, a dwa rodzaje linii
panel dodatkowo odczytuje:

  [stan] WŁĄCZONY: opis     pasek stanu (kolor wg pierwszego słowa)
  [pole] Nazwa: wartość     pole z bieżącą wartością
"""
import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

KATALOG_BOTOW = Path(__file__).resolve().parent / "boty"
KOLORY_STANU = {"WŁĄCZONY": "#2e8b57", "WSTRZYMANY": "#c98a00"}
KOLOR_NIEAKTYWNY = "#6b7280"
CZAS_NA_ZAMKNIECIE = 2.0  # [s] potem proces jest zamykany siłą
MAKS_LINII_DZIENNIKA = 2000


def wczytaj_boty(katalog=KATALOG_BOTOW):
    """Zwraca (lista botów, lista błędów); zepsuty plik nie blokuje pozostałych."""
    boty, bledy = [], []
    for plik in sorted(katalog.glob("*/*.json")):
        try:
            bot = json.loads(plik.read_text(encoding="utf-8"))
            bot["nazwa"], bot["skrypt"]
        except (OSError, ValueError, KeyError) as e:
            bledy.append(f"{plik}: {e}")
            continue
        bot["katalog"] = plik.parent
        boty.append(bot)
    return sorted(boty, key=lambda b: b["nazwa"]), bledy


class Panel(tk.Tk):
    def __init__(self, boty, bledy=()):
        super().__init__()
        self.title("Panel botów")
        self.geometry("980x660")
        self.minsize(900, 520)
        self.boty = boty
        self.proces = None
        self.linie = queue.Queue()
        self.pola_opcji = {}
        self.pola_stanu = {}
        self.przyciski_polecen = []

        ttk.Style(self).theme_use("clam")
        self.lista = tk.Listbox(self, width=26, exportselection=False, font=("Sans", 11))
        self.lista.pack(side="left", fill="y", padx=(10, 0), pady=10)
        for bot in boty:
            self.lista.insert("end", bot["nazwa"])
        self.lista.bind("<<ListboxSelect>>", self.wybrano_bota)

        prawa = ttk.Frame(self, padding=10)
        prawa.pack(side="left", fill="both", expand=True)

        self.tytul = ttk.Label(prawa, font=("Sans", 14, "bold"))
        self.tytul.pack(anchor="w")
        self.opis = ttk.Label(prawa, wraplength=680, justify="left")
        self.opis.pack(anchor="w", pady=(2, 8))
        self.formularz = ttk.Frame(prawa)
        self.formularz.pack(fill="x")

        przyciski = ttk.Frame(prawa)
        przyciski.pack(fill="x", pady=8)
        self.przycisk_start = ttk.Button(przyciski, text="Uruchom", command=self.uruchom)
        self.przycisk_start.pack(side="left")
        self.przycisk_stop = ttk.Button(przyciski, text="Zatrzymaj", command=self.zatrzymaj,
                                        state="disabled")
        self.przycisk_stop.pack(side="left", padx=(6, 12))
        self.polecenia = ttk.Frame(przyciski)
        self.polecenia.pack(side="left")

        self.stan = tk.Label(prawa, text="ZATRZYMANY", fg="white", bg=KOLOR_NIEAKTYWNY,
                             font=("Sans", 13, "bold"), pady=6)
        self.stan.pack(fill="x")
        self.ramka_pol = ttk.Frame(prawa)
        self.ramka_pol.pack(fill="x", pady=(6, 0))

        ramka_dziennika = ttk.Frame(prawa)
        ramka_dziennika.pack(fill="both", expand=True, pady=(8, 0))
        pasek = ttk.Scrollbar(ramka_dziennika)
        pasek.pack(side="right", fill="y")
        self.dziennik = tk.Text(ramka_dziennika, height=10, state="disabled", wrap="word",
                                font=("Monospace", 9), yscrollcommand=pasek.set)
        self.dziennik.pack(side="left", fill="both", expand=True)
        pasek.config(command=self.dziennik.yview)

        self.protocol("WM_DELETE_WINDOW", self.zamknij)
        for blad in bledy:
            self.dopisz(f"[panel] pominięto plik: {blad}")
        if boty:
            self.lista.selection_set(0)
            self.wybrano_bota()
        else:
            self.dopisz(f"[panel] brak botów w {KATALOG_BOTOW}")
            self.przycisk_start.config(state="disabled")
        self.zegar = self.after(100, self.odbieraj)

    @property
    def dziala(self):
        return self.proces is not None and self.proces.poll() is None

    @property
    def bot(self):
        wybor = self.lista.curselection()
        return self.boty[wybor[0]] if wybor else None

    def wybrano_bota(self, _zdarzenie=None):
        bot = self.bot
        if bot is None:
            return
        self.tytul.config(text=bot["nazwa"])
        self.opis.config(text=bot.get("opis", ""))
        for widget in self.formularz.winfo_children():
            widget.destroy()
        for widget in self.polecenia.winfo_children():
            widget.destroy()
        self.pola_opcji = {}
        self.przyciski_polecen = []

        for wiersz, opcja in enumerate(bot.get("opcje", ())):
            domyslnie = opcja.get("domyslnie")
            if opcja.get("typ") == "bool":
                zmienna = tk.BooleanVar(value=bool(domyslnie))
                ttk.Checkbutton(self.formularz, text=opcja["etykieta"],
                                variable=zmienna).grid(row=wiersz, column=0, columnspan=2,
                                                       sticky="w", pady=1)
            else:
                zmienna = tk.StringVar(value="" if domyslnie is None else str(domyslnie))
                ttk.Label(self.formularz, text=opcja["etykieta"]).grid(row=wiersz, column=0,
                                                                       sticky="w", padx=(0, 8))
                if opcja.get("typ") == "wybor":
                    pole = ttk.Combobox(self.formularz, textvariable=zmienna, width=8,
                                        state="readonly", values=opcja["wartosci"])
                    if "polecenie" in opcja:
                        pole.bind("<<ComboboxSelected>>", lambda _z, o=opcja, z=zmienna:
                                  self.wyslij(f"{o['polecenie']} {z.get()}"))
                else:
                    pole = ttk.Entry(self.formularz, textvariable=zmienna, width=10)
                pole.grid(row=wiersz, column=1, pady=1)
            self.pola_opcji[opcja["arg"]] = (opcja, zmienna)

        for wpis in bot.get("polecenia", ()):
            przycisk = ttk.Button(self.polecenia, text=wpis["etykieta"], state="disabled",
                                  command=lambda p=wpis["polecenie"]: self.wyslij(p))
            przycisk.pack(side="left", padx=2)
            self.przyciski_polecen.append(przycisk)

    def argumenty(self):
        """Buduje argumenty z formularza. Rzuca ValueError przy złej liczbie."""
        wynik = []
        for arg, (opcja, zmienna) in self.pola_opcji.items():
            typ = opcja.get("typ", "str")
            if typ == "bool":
                if zmienna.get():
                    wynik.append(arg)
                continue
            wartosc = zmienna.get().strip()
            if not wartosc:
                continue
            try:
                {"float": float, "int": int}.get(typ, str)(wartosc.replace(",", "."))
            except ValueError:
                raise ValueError(f"{opcja['etykieta']}: niepoprawna wartość „{wartosc}”")
            wynik += [arg, wartosc.replace(",", ".") if typ == "float" else wartosc]
        return wynik

    def uruchom(self):
        if self.dziala or self.bot is None:
            return
        bot = self.bot
        try:
            argumenty = self.argumenty()
        except ValueError as e:
            messagebox.showerror("Niepoprawne ustawienie", str(e), parent=self)
            return
        for widget in self.ramka_pol.winfo_children():
            widget.destroy()
        self.pola_stanu = {}
        self.dopisz(f"[panel] uruchamiam: {bot['nazwa']} {' '.join(argumenty)}")
        try:
            self.proces = subprocess.Popen(
                [sys.executable, "-u", bot["skrypt"], *argumenty], cwd=bot["katalog"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as e:
            self.dopisz(f"[panel] nie udało się uruchomić: {e}")
            self.proces = None
            return
        threading.Thread(target=self.czytaj_wyjscie, args=(self.proces,), daemon=True).start()
        self.ustaw_stan("URUCHAMIANIE")
        self.ustaw_przyciski(True)

    def czytaj_wyjscie(self, proces):
        for linia in proces.stdout:
            self.linie.put(linia.rstrip("\n"))
        self.linie.put(None)

    def odbieraj(self):
        try:
            while True:
                linia = self.linie.get_nowait()
                if linia is None:
                    self.proces_skonczony()
                else:
                    self.obsluz_linie(linia)
        except queue.Empty:
            pass
        self.zegar = self.after(100, self.odbieraj)

    def obsluz_linie(self, linia):
        if linia.startswith("[stan] "):
            self.ustaw_stan(linia[len("[stan] "):])
        elif linia.startswith("[pole] ") and ": " in linia:
            nazwa, wartosc = linia[len("[pole] "):].split(": ", 1)
            self.ustaw_pole(nazwa, wartosc)
            return  # częste odczyty tylko by zaśmiecały dziennik
        self.dopisz(linia)

    def ustaw_pole(self, nazwa, wartosc):
        if nazwa not in self.pola_stanu:
            ramka = ttk.Frame(self.ramka_pol)
            ramka.pack(side="left", padx=(0, 20))
            ttk.Label(ramka, text=f"{nazwa}:").pack(side="left")
            etykieta = ttk.Label(ramka, font=("Sans", 11, "bold"))
            etykieta.pack(side="left", padx=(4, 0))
            self.pola_stanu[nazwa] = etykieta
        self.pola_stanu[nazwa].config(text=wartosc)

    def ustaw_stan(self, tekst):
        kolor = KOLORY_STANU.get(tekst.split(":")[0].split()[0], KOLOR_NIEAKTYWNY)
        self.stan.config(text=tekst, bg=kolor)

    def ustaw_przyciski(self, dziala):
        self.przycisk_start.config(state="disabled" if dziala else "normal")
        self.przycisk_stop.config(state="normal" if dziala else "disabled")
        self.lista.config(state="disabled" if dziala else "normal")
        for przycisk in self.przyciski_polecen:
            przycisk.config(state="normal" if dziala else "disabled")

    def proces_skonczony(self):
        kod = self.proces.wait() if self.proces else 0
        self.proces = None
        self.ustaw_stan("ZATRZYMANY")
        self.ustaw_przyciski(False)
        self.dopisz(f"[panel] bot zakończył pracę (kod {kod})")

    def wyslij(self, polecenie):
        if not self.dziala:
            return
        try:
            self.proces.stdin.write(polecenie + "\n")
            self.proces.stdin.flush()
        except (OSError, ValueError):
            pass  # proces właśnie się kończy; panel dowie się o tym z wyjścia

    def zatrzymaj(self):
        if not self.dziala:
            return
        proces = self.proces
        self.wyslij("koniec")
        self.after(int(CZAS_NA_ZAMKNIECIE * 1000),
                   lambda: proces.poll() is None and proces.terminate())

    def dopisz(self, tekst):
        self.dziennik.config(state="normal")
        self.dziennik.insert("end", tekst + "\n")
        linii = int(self.dziennik.index("end-1c").split(".")[0])
        if linii > MAKS_LINII_DZIENNIKA:
            self.dziennik.delete("1.0", f"{linii - MAKS_LINII_DZIENNIKA}.0")
        self.dziennik.see("end")
        self.dziennik.config(state="disabled")

    def zamknij(self):
        if self.dziala:
            self.wyslij("koniec")
            try:
                self.proces.wait(timeout=CZAS_NA_ZAMKNIECIE)
            except subprocess.TimeoutExpired:
                self.proces.terminate()
        self.destroy()

    def destroy(self):
        self.after_cancel(self.zegar)
        super().destroy()


def main():
    Panel(*wczytaj_boty()).mainloop()


if __name__ == "__main__":
    main()
