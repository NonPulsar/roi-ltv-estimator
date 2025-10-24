import math
import tkinter as tk
from tkinter import ttk, messagebox

# -----------------------------
# Narzędzia pomocnicze
# -----------------------------

def parse_perc_list(s):
    """
    Parsuje listę procentów z tekstu (spacje/komy/średniki), np.
    '-85.1 -71.5 -61.9 -52.0 -40.7' -> [-0.851, -0.715, -0.619, -0.520, -0.407]
    Waliduje, że ROI > -100%.
    """
    if not s.strip():
        raise ValueError("Podaj 5 wartości ROI (D0, D3, D7, D14, D28).")
    cleaned = s.replace(",", ".").replace(";", " ")
    parts = [p for p in cleaned.replace("\t", " ").split() if p]
    if len(parts) != 5:
        raise ValueError("Oczekuję dokładnie 5 wartości ROI: D0, D3, D7, D14, D28.")
    vals = []
    for p in parts:
        v = float(p) / 100.0
        if v <= -1.0:
            raise ValueError("Każdy ROI musi być > -100%.")
        vals.append(v)
    return vals  # ułamki

def to_pct(x, nd=1):
    return f"{x*100:.{nd}f}%"

def r2_score(y_true, y_pred):
    """Klasyczny R^2."""
    ybar = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - ybar) ** 2 for a in y_true)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

# -----------------------------
# Modele ogona: power-law i log-poly(2)
# -----------------------------

def fit_power_law(one_plus, days):
    """
    Dopasowanie: log(1+ROI_t) = a + b*log(t+1)
    Zwraca (a,b), predykcję funkcji i R^2 w domenie log.
    """
    eps = 1e-12
    X = [math.log(d + 1.0) for d in days]
    Y = [math.log(max(eps, op)) for op in one_plus]
    # LS liniowe
    n = len(X)
    sx = sum(X); sy = sum(Y)
    sxx = sum(x*x for x in X); sxy = sum(x*y for x,y in zip(X,Y))
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("Degeneracja regresji (var(log(t+1))=0).")
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n

    def predict(t):
        return math.exp(a + b * math.log(t + 1.0))

    yhat = [math.log(max(eps, predict(d))) for d in days]
    r2 = r2_score(Y, yhat)
    return (a, b), predict, r2

def fit_log_poly2(one_plus, days):
    """
    Dopasowanie: log(1+ROI_t) = a + b*log(t+1) + c*[log(t+1)]^2
    Wygodne, bo to nadal LS liniowe w [1, X, X^2].
    """
    eps = 1e-12
    X = [math.log(d + 1.0) for d in days]
    X2 = [x*x for x in X]
    Y = [math.log(max(eps, op)) for op in one_plus]
    # LS dla trzech parametrów: [1, X, X^2]
    n = len(X)
    S1 = n
    Sx = sum(X)
    Sx2 = sum(X2)
    Sx3 = sum(x*x2 for x, x2 in zip(X, X2))
    Sx4 = sum(x2*x2 for x2 in X2)
    Sy = sum(Y)
    Sxy = sum(x*y for x,y in zip(X,Y))
    Sx2y = sum(x2*y for x2,y in zip(X2,Y))

    # Rozwiąż układ 3x3 (Gauss)
    # | S1  Sx  Sx2 | |a| = | Sy   |
    # | Sx  Sx2 Sx3 | |b|   | Sxy  |
    # | Sx2 Sx3 Sx4 | |c|   | Sx2y |
    A = [[S1, Sx, Sx2],
         [Sx, Sx2, Sx3],
         [Sx2, Sx3, Sx4]]
    B = [Sy, Sxy, Sx2y]

    # prosta eliminacja Gaussa
    for i in range(3):
        # pivot
        piv = A[i][i]
        if abs(piv) < 1e-12:
            raise ValueError("Degeneracja regresji (macierz osobliwa).")
        # normalizuj wiersz
        inv = 1.0 / piv
        for j in range(i, 3):
            A[i][j] *= inv
        B[i] *= inv
        # wyeliminuj kolumnę i w pozostałych wierszach
        for k in range(3):
            if k == i: continue
            f = A[k][i]
            for j in range(i, 3):
                A[k][j] -= f * A[i][j]
            B[k] -= f * B[i]

    a, b, c = B

    def predict(t):
        xt = math.log(t + 1.0)
        return math.exp(a + b * xt + c * xt * xt)

    yhat = [math.log(max(eps, predict(d))) for d in days]
    r2 = r2_score(Y, yhat)
    return (a, b, c), predict, r2

# -----------------------------
# Logika przeliczeń
# -----------------------------

def compute_required_curve_auto_tail(base_roi_5, target_roi_at_H, H, model="power"):
    """
    1) Fit model na punktach D0,D3,D7,D14,D28 dla (1+ROI).
    2) Wyznacz (1+ROI_H)_pred z modelu.
    3) f_t = (1+ROI_t_base) / (1+ROI_H_pred).
    4) Skala do celu: (1+ROI*_t) = f_t * (1+ROI_target_at_H). Zwraca listę ROI* (ułamki),
       (1+ROI_H_pred), dane o modelu i R^2.
    """
    days = [0, 3, 7, 14, 28]
    one_plus = [1.0 + r for r in base_roi_5]

    if model == "power":
        params, predictor, r2 = fit_power_law(one_plus, days)
    elif model == "logpoly2":
        params, predictor, r2 = fit_log_poly2(one_plus, days)
    else:
        raise ValueError("Nieznany model ogona.")

    # prognoza na H
    if H == 28:
        one_plus_H_pred = one_plus[-1]
    else:
        one_plus_H_pred = predictor(H)

    if one_plus_H_pred <= 0:
        raise ValueError("Przewidywane (1+ROI_H) nie może być <= 0.")

    f = [op / one_plus_H_pred for op in one_plus]
    one_plus_target = 1.0 + target_roi_at_H
    required_one_plus = [fi * one_plus_target for fi in f]
    required_roi = [x - 1.0 for x in required_one_plus]
    return required_roi, one_plus_H_pred, (model, params), r2, predictor

def apply_organic_uplift_on_revenue(required_roi_curve, uplift_share):
    """ROI_org = (1 + ROI_bez_org) * (1 + uplift) - 1"""
    u = uplift_share
    return [(1.0 + roi) * (1.0 + u) - 1.0 for roi in required_roi_curve]

# -----------------------------
# GUI
# -----------------------------

class ROIEstimatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Estimator ROI: D0..D28 → LTV@D28/D90/D360 • organika: on/off")
        self.root.geometry("1060x780")

        # Ramki
        input_frame = ttk.LabelFrame(root, text="Dane wejściowe")
        input_frame.pack(fill="x", padx=12, pady=8)

        options_frame = ttk.LabelFrame(root, text="Ustawienia celu i modelu ogona")
        options_frame.pack(fill="x", padx=12, pady=8)

        output_frame = ttk.LabelFrame(root, text="Wymagane ROI (D0, D3, D7, D14, D28)")
        output_frame.pack(fill="both", expand=True, padx=12, pady=8)

        ltv_frame = ttk.LabelFrame(root, text="Prognozowane LTV (z samej bazy) i podsumowanie")
        ltv_frame.pack(fill="both", expand=False, padx=12, pady=8)

        # --- Dane wejściowe
        ttk.Label(input_frame, text="Bazowe ROI (D0 D3 D7 D14 D28) w %:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.entry_base = ttk.Entry(input_frame, width=60)
        self.entry_base.grid(row=0, column=1, padx=8, pady=6)
        self.entry_base.insert(0, "-85.1 -71.5 -61.9 -52.0 -40.7")

        ttk.Label(input_frame, text="Docelowe LTV% (na wybrany horyzont):").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.entry_target = ttk.Entry(input_frame, width=20)
        self.entry_target.grid(row=1, column=1, sticky="w", padx=8, pady=6)
        self.entry_target.insert(0, "20")

        ttk.Label(input_frame, text="Organiczny uplift przychodu (opcjonalnie, %):").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        self.entry_org = ttk.Entry(input_frame, width=20)
        self.entry_org.grid(row=2, column=1, sticky="w", padx=8, pady=6)
        self.entry_org.insert(0, "")

        # --- Ustawienia celu i modelu
        self.horizon = tk.IntVar(value=90)  # domyślnie D90
        rb28 = ttk.Radiobutton(options_frame, text="Horyzont LTV: D28", variable=self.horizon, value=28)
        rb90 = ttk.Radiobutton(options_frame, text="Horyzont LTV: D90", variable=self.horizon, value=90)
        rb360 = ttk.Radiobutton(options_frame, text="Horyzont LTV: D360", variable=self.horizon, value=360)
        rb28.grid(row=0, column=0, padx=8, pady=6, sticky="w")
        rb90.grid(row=0, column=1, padx=8, pady=6, sticky="w")
        rb360.grid(row=0, column=2, padx=8, pady=6, sticky="w")

        ttk.Label(options_frame, text="Model ogona:").grid(row=0, column=3, padx=8, pady=6, sticky="e")
        self.model_choice = tk.StringVar(value="power")
        model_cb = ttk.Combobox(options_frame, textvariable=self.model_choice, state="readonly",
                                values=["power", "logpoly2"], width=10)
        model_cb.grid(row=0, column=4, padx=8, pady=6, sticky="w")
        model_cb.tooltip = "power = log-lin; logpoly2 = log-kwadrat (bardziej elastyczny)"

        # Przełącznik: czy cel LTV uwzględnia organikę?
        self.target_includes_org = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(options_frame,
                              text="Docelowe LTV uwzględnia organikę (uplift)?",
                              variable=self.target_includes_org)
        chk.grid(row=1, column=0, columnspan=3, padx=8, pady=6, sticky="w")

        self.btn_calc = ttk.Button(options_frame, text="Oblicz", command=self.calculate)
        self.btn_calc.grid(row=1, column=4, padx=8, pady=6, sticky="e")

        # --- Tabela wyników (wymagane ROI)
        cols = ("Dzień",
                "Wymagany ROI (paid-only)",
                "ROI po dodaniu organiki (+% przychodu)")
        self.tree = ttk.Treeview(output_frame, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=280 if c != "Dzień" else 80)
        self.tree.pack(fill="both", expand=True, padx=8, pady=8)

        # --- LTV i podsumowanie
        self.ltv_label = ttk.Label(ltv_frame, text="", anchor="w", justify="left")
        self.ltv_label.pack(fill="x", padx=8, pady=6)

        self.summary = tk.Text(ltv_frame, height=9)
        self.summary.pack(fill="x", padx=8, pady=6)

    def calculate(self):
        # wyczyść
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.summary.delete("1.0", tk.END)
        self.ltv_label.config(text="")

        try:
            base = parse_perc_list(self.entry_base.get())
            target_pct = float(self.entry_target.get().replace(",", "."))
            target_roi_input = target_pct / 100.0

            org_text = (self.entry_org.get() or "").strip()
            uplift = float(org_text.replace(",", "."))/100.0 if org_text else 0.0
            if uplift < 0:
                raise ValueError("Uplift organiki nie może być ujemny.")

            H = self.horizon.get()
            model = self.model_choice.get()

            # --- Prognoza LTV D90/D360 z samej bazy (niezależnie od celu)
            days_base = [0, 3, 7, 14, 28]
            one_plus_base = [1.0 + r for r in base]
            if model == "power":
                (_, _), predictor, r2_fit = fit_power_law(one_plus_base, days_base)
                model_name = "power"
            else:
                (_, _, _), predictor, r2_fit = fit_log_poly2(one_plus_base, days_base)
                model_name = "logpoly2"

            # prognoza (1+ROI) dla 90 i 360
            one_plus_28 = one_plus_base[-1]
            one_plus_90 = predictor(90)
            one_plus_360 = predictor(360)
            base_pred_roi_28 = one_plus_28 - 1.0
            base_pred_roi_90 = one_plus_90 - 1.0
            base_pred_roi_360 = one_plus_360 - 1.0

            # --- Interpretacja celu LTV (paid-only vs z organiką)
            if self.target_includes_org.get():
                # Cel dotyczy TOTAL (paid + uplift). Chcemy dobrać paid tak, by po dodaniu uplifta wyjść na target.
                # (1+ROI_paid_target_at_H) * (1+uplift) = (1 + ROI_total_target_at_H)
                one_plus_target_paid = (1.0 + target_roi_input) / (1.0 + uplift if uplift > 0 else 1.0)
                target_roi_paid_at_H = one_plus_target_paid - 1.0
            else:
                # Cel dotyczy paid-only (bez organiki)
                target_roi_paid_at_H = target_roi_input

            # --- Krzywa wymagana (paid-only) dla celu na H
            required_paid, one_plus_H_pred, (model_used, params), r2_model, _ = compute_required_curve_auto_tail(
                base_roi_5=base,
                target_roi_at_H=target_roi_paid_at_H,
                H=H,
                model=model
            )

            # --- Kolumna „po organice” (TOTAL)
            required_total = apply_organic_uplift_on_revenue(required_paid, uplift) if uplift > 0 else None

            # --- Wypełnij tabelę
            days = ["D0", "D3", "D7", "D14", "D28"]
            for idx, d in enumerate(days):
                row = [d, to_pct(required_paid[idx])]
                row.append(to_pct(required_total[idx]) if required_total is not None else "—")
                self.tree.insert("", "end", values=tuple(row))

            # --- Sekcja LTV z bazy
            ltv_txt = (
                f"Prognozowane LTV (z krzywej bazowej, model: {model_name}):   "
                f"D28: {to_pct(base_pred_roi_28)}   •   D90: {to_pct(base_pred_roi_90)}   •   D360: {to_pct(base_pred_roi_360)}"
            )
            self.ltv_label.config(text=ltv_txt)

            # --- Podsumowanie
            lines = []
            lines.append(f"Horyzont docelowy: D{H}")
            if self.target_includes_org.get():
                lines.append(f"Docelowe LTV% (Z ORGANIKĄ): {to_pct(target_roi_input)}")
                if uplift > 0:
                    lines.append(f"Uplift organiki użyty przy celu: {to_pct(uplift)}")
                    lines.append(f"Z tego wynika cel paid-only: {to_pct(target_roi_paid_at_H)} (czyli (1+target_total)/(1+uplift) − 1)")
                else:
                    lines.append("Uwaga: zaznaczono 'z organiką', ale uplift = 0% — cel paid = cel total.")
            else:
                lines.append(f"Docelowe LTV% (PAID-ONLY): {to_pct(target_roi_input)}")
                if uplift > 0:
                    lines.append(f"Uplift organiki dodany *po* wyliczeniu paid: {to_pct(uplift)}")

            lines.append(f"Przewidywane (1 + ROI_D{H}) z modelu (na bazie): {one_plus_H_pred:.4f}  →  ROI_D{H} ≈ {to_pct(one_plus_H_pred - 1.0)}")
            # model params / R^2
            if model_used == "power":
                a, b = params
                lines.append(f"Model ogona: power  →  log(1+ROI) = {a:.4f} + {b:.4f}·log(t+1)")
            else:
                a, b, c = params
                lines.append(f"Model ogona: logpoly2  →  log(1+ROI) = {a:.4f} + {b:.4f}·log(t+1) + {c:.4f}·[log(t+1)]²")
            lines.append(f"Dopasowanie (R² w domenie log): {r2_model:.4f}")

            if uplift > 0:
                lines.append("Transformacja TOTAL: ROI_total = (1 + ROI_paid) × (1 + uplift) − 1")
            else:
                lines.append("Organika: brak zastosowanego uplifta (0%).")

            self.summary.insert("1.0", "\n".join(lines))

        except Exception as e:
            messagebox.showerror("Błąd", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ROIEstimatorGUI(root)
    root.mainloop()
