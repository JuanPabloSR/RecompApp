import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import sqlite3
import logging
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ─── Logging ─────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RecompApp")

# ─── Constantes ──────────────────────────────────
EXPECTED_COLUMNS = ["Date", "Protein", "Fats", "Sat Fats", "Carbs", "Sugars", "Calories", "Weight", "Body Fat"]
MAX_ROWS_DISPLAY = 100
ML_MODELS = {
    "Regresión Lineal": LinearRegression,
    "SVM": SVR,
    "Árbol de Decisión": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "KNN": KNeighborsRegressor,
}

# ─── Fallout Design System ───────────────────────
FO_BG      = "#050a05"   # fondo principal (negro verdoso)
FO_GREEN   = "#4af626"   # verde fósforo brillante
FO_DIM     = "#1a3a1a"   # verde muy oscuro (bordes sutiles)
FO_MID     = "#0d6b06"   # verde medio (texto tabs inactivos, visible)
FO_HOVER   = "#16a308"   # verde hover (fondo al pasar mouse)
FO_BLACK   = "#000000"   # negro absoluto
FO_FONT    = "Consolas"  # fuente monoespaciada global
FO_FONT_B  = ("Consolas", 12, "bold")
FO_FONT_N  = ("Consolas", 12)
FO_FONT_SM = ("Consolas", 10)
FO_FONT_LG = ("Consolas", 14, "bold")
FO_FONT_XL = ("Consolas", 16, "bold")


class RecompApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.df: pd.DataFrame | None = None
        self.modelo_actual = None
        self.columnas_x_entrenadas: list[str] = []
        # Typewriter animation state
        self._is_typing = False
        self._typing_queue: list[str] = []
        self._setup_window()
        self._setup_styles()
        self._build_ui()

    # ──────────────────────────────────────────────
    # Configuración de Ventana
    # ──────────────────────────────────────────────
    def _setup_window(self):
        self.title("RECOMP MASTER PRO // TERMINAL v4.0")
        self.geometry("1200x800")
        self.minsize(900, 600)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        self.configure(fg_color=FO_BG)

    # ──────────────────────────────────────────────
    # Estilo ttk (Treeview Fallout)
    # ──────────────────────────────────────────────
    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Dark.Treeview",
            background=FO_BLACK, foreground=FO_GREEN, fieldbackground=FO_BLACK,
            rowheight=30, font=FO_FONT_N, borderwidth=1, relief="flat",
        )
        style.configure(
            "Dark.Treeview.Heading",
            background=FO_BG, foreground=FO_GREEN,
            font=FO_FONT_B, borderwidth=1, relief="flat",
        )
        style.map("Dark.Treeview",
                  background=[("selected", FO_GREEN)],
                  foreground=[("selected", FO_BLACK)])
        style.map("Dark.Treeview.Heading",
                  background=[("active", FO_DIM)])

    # ──────────────────────────────────────────────
    # Construcción de la UI
    # ──────────────────────────────────────────────
    def _build_ui(self):
        try:
            self.tabview = ctk.CTkTabview(
                self, corner_radius=4,
                fg_color=FO_BG, border_width=1, border_color=FO_GREEN,
                segmented_button_selected_color=FO_GREEN,
                segmented_button_unselected_color=FO_BG,
                segmented_button_selected_hover_color=FO_HOVER,
                segmented_button_unselected_hover_color=FO_DIM,
            )
            self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

            self.tab_datos    = self.tabview.add("[DATOS]")
            self.tab_gestion  = self.tabview.add("[GESTION]")
            self.tab_graficos = self.tabview.add("[GRAFICOS]")
            self.tab_ml       = self.tabview.add("[ML ENGINE]")

            # Fuente terminal para las pestañas — texto visible en inactivos
            try:
                self.tabview._segmented_button.configure(
                    font=ctk.CTkFont(family=FO_FONT, size=13, weight="bold"),
                    selected_color=FO_GREEN, unselected_color=FO_BG,
                    text_color=FO_BLACK, unselected_hover_color=FO_DIM,
                    text_color_disabled=FO_DIM,
                )
                # Forzar texto verde medio en pestañas no seleccionadas
                for btn in self.tabview._segmented_button._buttons_dict.values():
                    btn.configure(text_color=FO_MID, text_color_disabled=FO_MID)
            except Exception:
                pass

            self._build_tab_datos()
            self._build_tab_gestion()
            self._build_tab_graficos()
            self._build_tab_ml()

            logger.info("Interfaz inicializada correctamente.")
        except Exception as e:
            logger.error(f"Error construyendo la UI: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"No se pudo inicializar la interfaz:\n{e}")

    # ── Helper: Fallout Button ──
    def _fo_btn(self, parent, text, command, **kw):
        return ctk.CTkButton(
            parent, text=text, command=command,
            fg_color=FO_BLACK, hover_color=FO_HOVER,
            border_width=1, border_color=FO_GREEN,
            text_color=FO_GREEN, corner_radius=4,
            font=ctk.CTkFont(family=FO_FONT, size=12, weight="bold"),
            **kw,
        )

    # ── Helper: Fallout Label ──
    def _fo_label(self, parent, text, size=12, **kw):
        return ctk.CTkLabel(
            parent, text=text, text_color=FO_GREEN,
            font=ctk.CTkFont(family=FO_FONT, size=size, weight="bold"),
            **kw,
        )

    # ── Helper: Fallout Entry ──
    def _fo_entry(self, parent, placeholder="", **kw):
        return ctk.CTkEntry(
            parent, fg_color=FO_BLACK, text_color=FO_GREEN,
            border_width=1, border_color=FO_GREEN, corner_radius=4,
            placeholder_text=placeholder, placeholder_text_color=FO_DIM,
            font=ctk.CTkFont(family=FO_FONT, size=11),
            **kw,
        )

    # ── Helper: Fallout OptionMenu ──
    def _fo_option(self, parent, values, width=210):
        return ctk.CTkOptionMenu(
            parent, values=values, width=width,
            fg_color=FO_BLACK, button_color=FO_DIM, button_hover_color=FO_HOVER,
            text_color=FO_GREEN, dropdown_fg_color=FO_BLACK,
            dropdown_text_color=FO_GREEN, dropdown_hover_color=FO_DIM,
            font=ctk.CTkFont(family=FO_FONT, size=11),
        )

    # ── Helper: Fallout Frame (card) ──
    def _fo_card(self, parent, **kw):
        return ctk.CTkFrame(
            parent, fg_color=FO_BG, corner_radius=4,
            border_width=1, border_color=FO_GREEN, **kw,
        )

    # ══════════════════════════════════════════════
    # Tab 1 — DATOS
    # ══════════════════════════════════════════════
    def _build_tab_datos(self):
        top = ctk.CTkFrame(self.tab_datos, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=(10, 5))

        self._fo_btn(top, "> CARGAR DATASET", self._on_load_dataset,
                     width=200, height=36).pack(side="left", padx=(0, 15))

        self.file_status_label = ctk.CTkLabel(
            top, text="[SIN ARCHIVO]", text_color=FO_DIM,
            font=ctk.CTkFont(family=FO_FONT, size=11),
        )
        self.file_status_label.pack(side="left", fill="y")
        self._loaded_file_name = ""

        tree_frame = ctk.CTkFrame(self.tab_datos, fg_color="transparent")
        tree_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        scroll_y = ttk.Scrollbar(tree_frame, orient="vertical")
        scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
        self.treeview = ttk.Treeview(
            tree_frame, style="Dark.Treeview",
            yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set, show="headings",
        )
        scroll_y.config(command=self.treeview.yview)
        scroll_x.config(command=self.treeview.xview)

        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.treeview.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")

    def _on_load_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar dataset",
            filetypes=[
                ("Archivos CSV", "*.csv"),
                ("Archivos Excel", "*.xlsx *.xls"),
                ("Bases de datos SQLite", "*.db *.sqlite"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not file_path:
            return
        file_name = os.path.basename(file_path)
        logger.info(f"Archivo seleccionado: {file_path}")
        try:
            ext = os.path.splitext(file_name)[1].lower()
            if ext == ".csv":
                try:
                    self.df = pd.read_csv(file_path, sep=",")
                    if self.df.shape[1] <= 1:
                        self.df = pd.read_csv(file_path, sep=";")
                except Exception:
                    self.df = pd.read_csv(file_path, sep=";")
            elif ext in (".xlsx", ".xls"):
                self.df = pd.read_excel(file_path)
            elif ext in (".db", ".sqlite"):
                conn = sqlite3.connect(file_path)
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
                if tables.empty:
                    conn.close()
                    raise ValueError("La base de datos no contiene tablas.")
                table_name = tables.iloc[0, 0]
                self.df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
                conn.close()
                logger.info(f"Tabla SQLite cargada: '{table_name}'")
            else:
                raise ValueError(f"Formato no soportado: {ext}")

            logger.info(f"Dataset cargado: {self.df.shape[0]} filas × {self.df.shape[1]} columnas")
            self._loaded_file_name = file_name
            self._update_file_status()
            self.update_treeview()
            self.actualizar_dropdowns_graficos()
            self.actualizar_opciones_ml()

        except Exception as ex:
            logger.error(f"Error al cargar archivo: {ex}")
            logger.error(traceback.format_exc())
            self.df = None
            self._loaded_file_name = ""
            self.file_status_label.configure(text=f"[ERROR] {file_name}", text_color="#ff3333")
            messagebox.showerror("Error de carga", f"No se pudo leer el archivo:\n{ex}")

    def _update_file_status(self):
        if self.df is None or not self._loaded_file_name:
            return
        total = self.df.shape[0]
        display = min(total, MAX_ROWS_DISPLAY)
        if total > MAX_ROWS_DISPLAY:
            text = f"[OK] {self._loaded_file_name} // {display}/{total} filas"
        else:
            text = f"[OK] {self._loaded_file_name} // {total} filas x {self.df.shape[1]} cols"
        self.file_status_label.configure(text=text, text_color=FO_GREEN)

    def update_treeview(self):
        if self.df is None or self.df.empty:
            return
        self.treeview.delete(*self.treeview.get_children())
        display_df = self.df.head(MAX_ROWS_DISPLAY)
        cols = list(display_df.columns)
        self.treeview["columns"] = cols
        for col in cols:
            self.treeview.heading(
                col, text=col, anchor="center",
                command=lambda c=col: self.sort_treeview(c, False),
            )
            self.treeview.column(col, anchor="center", width=110, minwidth=80)
        for _, row in display_df.iterrows():
            self.treeview.insert("", "end", values=[str(v) for v in row])
        self._update_file_status()

    def sort_treeview(self, col: str, reverse: bool):
        try:
            data = [(self.treeview.set(child, col), child)
                    for child in self.treeview.get_children("")]
            try:
                data.sort(key=lambda t: float(t[0]), reverse=reverse)
            except (ValueError, TypeError):
                data.sort(key=lambda t: t[0].lower(), reverse=reverse)

            for index, (_, child) in enumerate(data):
                self.treeview.move(child, "", index)

            for c in self.treeview["columns"]:
                clean = c.replace(" ▲", "").replace(" ▼", "")
                if c == col:
                    ind = " ▼" if reverse else " ▲"
                    self.treeview.heading(c, text=clean + ind,
                                         command=lambda c=col: self.sort_treeview(c, not reverse))
                else:
                    self.treeview.heading(c, text=clean,
                                         command=lambda c=c: self.sort_treeview(c, False))
        except Exception as e:
            logger.error(f"Error ordenando Treeview: {e}")

    # ══════════════════════════════════════════════
    # Tab 2 — GESTION
    # ══════════════════════════════════════════════
    def _build_tab_gestion(self):
        main_frame = ctk.CTkFrame(self.tab_gestion, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # ── Panel Izquierdo ──
        left = ctk.CTkFrame(main_frame, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        # Tarjeta: Limpieza
        card_clean = self._fo_card(left)
        card_clean.pack(fill="x", padx=2, pady=(0, 6))
        self._fo_label(card_clean, "// LIMPIEZA DE DATOS", 14).pack(pady=(12, 8))

        for text, cmd in [
            ("> DETECTAR NULOS", self._detect_nulls),
            ("> ELIMINAR DUPLICADOS", self._remove_duplicates),
            ("> ELIMINAR NULOS [dropna]", self._drop_nulls),
            ("> IMPUTAR NULOS [MEDIA]", self._impute_nulls),
            ("> IMPUTAR NULOS [MODA]", self._impute_nulls_mode),
        ]:
            self._fo_btn(card_clean, text, cmd, width=250, height=32).pack(pady=2)
        # spacer bottom
        ctk.CTkFrame(card_clean, height=10, fg_color="transparent").pack()

        # Tarjeta: CRUD
        card_crud = self._fo_card(left)
        card_crud.pack(fill="x", padx=2, pady=(0, 6))
        self._fo_label(card_crud, "// REGISTRO MANUAL", 14).pack(pady=(12, 8))

        grid = ctk.CTkFrame(card_crud, fg_color="transparent")
        grid.pack(fill="x", padx=12, pady=(0, 5))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        fields = [
            ("Date", "entry_date", 0, 0),
            ("Calories", "entry_calories", 0, 1),
            ("Protein", "entry_protein", 1, 0),
            ("Weight", "entry_weight", 1, 1),
        ]
        self.crud_entries: dict[str, ctk.CTkEntry] = {}
        for lbl, attr, r, c in fields:
            cell = ctk.CTkFrame(grid, fg_color="transparent")
            cell.grid(row=r, column=c, padx=3, pady=2, sticky="ew")
            self._fo_label(cell, lbl, 10).pack(anchor="w")
            entry = self._fo_entry(cell, placeholder=lbl, height=28)
            entry.pack(fill="x")
            self.crud_entries[attr] = entry

        btns = ctk.CTkFrame(card_crud, fg_color="transparent")
        btns.pack(fill="x", padx=12, pady=(6, 12))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        btns.columnconfigure(2, weight=1)

        self._fo_btn(btns, "+ AÑADIR", self._add_record, height=32).grid(row=0, column=0, padx=(0, 2), sticky="ew")
        self._fo_btn(btns, "~ MODIFICAR", self._modify_record, height=32).grid(row=0, column=1, padx=2, sticky="ew")
        self._fo_btn(btns, "x ELIMINAR", self._delete_record, height=32).grid(row=0, column=2, padx=(2, 0), sticky="ew")

        # ── Panel Derecho — Consola ──
        right = self._fo_card(main_frame)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        self._fo_label(right, "// CONSOLA DE OPERACIONES", 14).pack(pady=(12, 8))

        self.console_log = ctk.CTkTextbox(
            right, corner_radius=4,
            fg_color=FO_BLACK, text_color=FO_GREEN,
            font=ctk.CTkFont(family=FO_FONT, size=12),
            border_width=1, border_color=FO_GREEN,
            state="disabled",
        )
        self.console_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._log_to_console("SISTEMA LISTO. CARGUE UN DATASET PARA COMENZAR.")

    # ── Helpers Gestión ──
    def _log_to_console(self, message: str):
        """Encola un mensaje para la animación typewriter estilo Fallout."""
        ts = datetime.now().strftime("%H:%M:%S")
        full_text = f"[{ts}] > {message}\n"
        self._typing_queue.append(full_text)
        if not self._is_typing:
            self._process_typing_queue()

    def _process_typing_queue(self):
        """Consume la cola de mensajes, animando uno a la vez."""
        if not self._typing_queue:
            self._is_typing = False
            return
        self._is_typing = True
        text = self._typing_queue.pop(0)
        self._typewrite(text, 0)

    def _typewrite(self, text: str, idx: int):
        """Inserta caracteres uno a uno con .after() — efecto typewriter."""
        if idx < len(text):
            # Insertar bloque de 2-3 chars para fluidez
            chunk = text[idx:idx + 3]
            self.console_log.configure(state="normal")
            self.console_log.insert("end", chunk)
            self.console_log.see("end")
            self.console_log.configure(state="disabled")
            self.after(18, self._typewrite, text, idx + 3)
        else:
            # Mensaje terminado, procesar siguiente de la cola
            self.after(60, self._process_typing_queue)

    def _check_df_loaded(self) -> bool:
        if self.df is None or self.df.empty:
            messagebox.showerror("SIN DATOS", "No hay un dataset cargado.\nVe a [DATOS] y carga un archivo primero.")
            return False
        return True

    def _detect_nulls(self):
        if not self._check_df_loaded():
            return
        try:
            null_counts = self.df.isnull().sum()
            total = null_counts.sum()
            if total == 0:
                self._log_to_console("SCAN COMPLETO: 0 valores nulos detectados.")
            else:
                report = f"ALERTA: {total} valor(es) nulo(s) detectados:\n"
                for col, count in null_counts.items():
                    if count > 0:
                        report += f"  [{col}]: {count} nulo(s)\n"
                self._log_to_console(report)
            logger.info(f"Detección de nulos: {total} total")
        except Exception as e:
            logger.error(f"Error detectando nulos: {e}")
            self._log_to_console(f"ERROR: {e}")

    def _remove_duplicates(self):
        if not self._check_df_loaded(): return
        try:
            before = self.df.shape[0]
            self.df.drop_duplicates(inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            removed = before - self.df.shape[0]
            msg = f"ELIMINADOS {removed} registro(s) duplicado(s)." if removed > 0 else "0 duplicados encontrados."
            self._log_to_console(msg); logger.info(msg); self.update_treeview()
        except Exception as e:
            logger.error(f"Error: {e}"); self._log_to_console(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))

    def _drop_nulls(self):
        if not self._check_df_loaded(): return
        try:
            before = self.df.shape[0]
            self.df.dropna(inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            removed = before - self.df.shape[0]
            msg = f"ELIMINADAS {removed} fila(s) con nulos." if removed > 0 else "0 filas con nulos."
            self._log_to_console(msg); logger.info(msg); self.update_treeview()
        except Exception as e:
            logger.error(f"Error: {e}"); self._log_to_console(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))

    def _impute_nulls(self):
        if not self._check_df_loaded(): return
        try:
            nc = self.df.select_dtypes(include="number").columns.tolist()
            nb = self.df[nc].isnull().sum().sum()
            if nb == 0:
                self._log_to_console("0 nulos en columnas numéricas."); return
            self.df[nc] = self.df[nc].fillna(self.df[nc].mean())
            self._log_to_console(f"IMPUTADOS {nb} nulo(s) con MEDIA. Cols: {', '.join(nc)}")
            logger.info(f"Imputados {nb} con media"); self.update_treeview()
        except Exception as e:
            logger.error(f"Error: {e}"); self._log_to_console(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))

    def _impute_nulls_mode(self):
        if not self._check_df_loaded(): return
        try:
            nb = self.df.isnull().sum().sum()
            if nb == 0:
                self._log_to_console("0 nulos detectados."); return
            for col in self.df.columns:
                if self.df[col].isnull().any():
                    m = self.df[col].mode()
                    if not m.empty:
                        self.df[col].fillna(m.iloc[0], inplace=True)
            imp = nb - self.df.isnull().sum().sum()
            self._log_to_console(f"IMPUTADOS {imp} nulo(s) con MODA.")
            logger.info(f"Imputados {imp} con moda"); self.update_treeview()
        except Exception as e:
            logger.error(f"Error: {e}"); self._log_to_console(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))

    def _get_crud_values(self):
        d = self.crud_entries["entry_date"].get().strip()
        c = self.crud_entries["entry_calories"].get().strip()
        p = self.crud_entries["entry_protein"].get().strip()
        w = self.crud_entries["entry_weight"].get().strip()
        if not all([d, c, p, w]):
            messagebox.showerror("CAMPOS VACÍOS", "Todos los campos son obligatorios."); return None
        try:
            return {"Date": d, "Calories": float(c), "Protein": float(p), "Weight": float(w)}
        except ValueError:
            messagebox.showerror("VALOR INVÁLIDO", "Calories, Protein y Weight deben ser numéricos."); return None

    def _clear_crud_entries(self):
        for e in self.crud_entries.values(): e.delete(0, "end")

    def _add_record(self):
        if not self._check_df_loaded(): return
        v = self._get_crud_values()
        if v is None: return
        try:
            nr = {col: None for col in self.df.columns}; nr.update(v)
            self.df = pd.concat([self.df, pd.DataFrame([nr])], ignore_index=True)
            self._clear_crud_entries()
            self._log_to_console(f"REGISTRO AÑADIDO: {v}"); logger.info(f"Add: {v}"); self.update_treeview()
        except Exception as e:
            logger.error(f"Error: {e}"); self._log_to_console(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))

    def _modify_record(self):
        if not self._check_df_loaded(): return
        sel = self.treeview.selection()
        if not sel:
            messagebox.showerror("SIN SELECCIÓN", "Seleccione una fila en [DATOS]."); return
        v = self._get_crud_values()
        if v is None: return
        try:
            idx = self.treeview.index(sel[0])
            for k, val in v.items():
                if k in self.df.columns: self.df.at[idx, k] = val
            self._clear_crud_entries()
            self._log_to_console(f"FILA {idx} MODIFICADA: {v}"); logger.info(f"Mod {idx}: {v}"); self.update_treeview()
        except Exception as e:
            logger.error(f"Error: {e}"); self._log_to_console(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))

    def _delete_record(self):
        if not self._check_df_loaded(): return
        sel = self.treeview.selection()
        if not sel:
            messagebox.showerror("SIN SELECCIÓN", "Seleccione fila(s) en [DATOS]."); return
        try:
            indices = [self.treeview.index(i) for i in sel]
            self.df.drop(index=indices, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self._log_to_console(f"ELIMINADAS {len(indices)} fila(s): {indices}")
            logger.info(f"Del: {indices}"); self.update_treeview()
        except Exception as e:
            logger.error(f"Error: {e}"); self._log_to_console(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))

    # ══════════════════════════════════════════════
    # Tab 3 — GRAFICOS
    # ══════════════════════════════════════════════
    def _build_tab_graficos(self):
        mf = ctk.CTkFrame(self.tab_graficos, fg_color="transparent")
        mf.pack(fill="both", expand=True, padx=5, pady=5)
        mf.columnconfigure(0, weight=0)
        mf.columnconfigure(1, weight=1)
        mf.rowconfigure(0, weight=1)

        ctrl = self._fo_card(mf, width=250)
        ctrl.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        ctrl.grid_propagate(False)

        self._fo_label(ctrl, "// CONTROLES", 14).pack(pady=(12, 10))

        self._fo_label(ctrl, "Variable X:", 10).pack(padx=12, anchor="w")
        self.opt_var_x = self._fo_option(ctrl, ["-- Cargar datos --"], 220)
        self.opt_var_x.pack(padx=12, pady=(2, 8))

        self._fo_label(ctrl, "Variable Y:", 10).pack(padx=12, anchor="w")
        self.opt_var_y = self._fo_option(ctrl, ["-- Cargar datos --"], 220)
        self.opt_var_y.pack(padx=12, pady=(2, 8))

        self._fo_label(ctrl, "Tipo Grafico:", 10).pack(padx=12, anchor="w")
        self.opt_chart_type = self._fo_option(ctrl, ["Dispersión (Scatter)", "Barras (Bar)", "Pastel (Pie)"], 220)
        self.opt_chart_type.pack(padx=12, pady=(2, 15))

        self._fo_btn(ctrl, "> GENERAR GRAFICO", self._generate_chart,
                     width=220, height=36).pack(pady=(0, 15))

        canvas_panel = self._fo_card(mf)
        canvas_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        self._fo_label(canvas_panel, "// VISUALIZACION", 14).pack(pady=(12, 5))

        plt.style.use("dark_background")
        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor=FO_BLACK)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(FO_BG)
        self.ax.text(0.5, 0.5, "SELECCIONE VARIABLES Y GENERE UN GRAFICO",
                     ha="center", va="center", fontsize=11, color=FO_DIM,
                     transform=self.ax.transAxes, family=FO_FONT)
        self._style_ax_fallout(self.ax)

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=canvas_panel)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _style_ax_fallout(self, ax):
        """Apply Fallout green styling to a matplotlib Axes."""
        ax.tick_params(colors=FO_GREEN, which="both")
        for spine in ax.spines.values():
            spine.set_color(FO_GREEN)

    def actualizar_dropdowns_graficos(self):
        if self.df is None or self.df.empty: return
        cols = list(self.df.columns)
        self.opt_var_x.configure(values=cols)
        self.opt_var_y.configure(values=cols)
        if len(cols) >= 2:
            self.opt_var_x.set(cols[0]); self.opt_var_y.set(cols[1])

    def _generate_chart(self):
        if not self._check_df_loaded(): return
        cx, cy = self.opt_var_x.get(), self.opt_var_y.get()
        ct = self.opt_chart_type.get()
        if cx not in self.df.columns:
            messagebox.showerror("Error", f"Columna '{cx}' no existe."); return
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_facecolor(FO_BG)
            self._style_ax_fallout(ax)

            if ct == "Dispersión (Scatter)":
                if not pd.api.types.is_numeric_dtype(self.df[cx]):
                    messagebox.showerror("Error", f"'{cx}' no es numérica."); return
                if cy not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[cy]):
                    messagebox.showerror("Error", f"'{cy}' no es numérica."); return
                ax.scatter(self.df[cx], self.df[cy], c=FO_GREEN, edgecolors=FO_HOVER, alpha=0.85, s=50)
                ax.set_xlabel(cx, color=FO_GREEN, fontsize=10, family=FO_FONT)
                ax.set_ylabel(cy, color=FO_GREEN, fontsize=10, family=FO_FONT)
                ax.set_title(f"{cy} vs {cx}", color=FO_GREEN, fontsize=12, weight="bold", family=FO_FONT)

            elif ct == "Barras (Bar)":
                if not pd.api.types.is_numeric_dtype(self.df[cy]):
                    messagebox.showerror("Error", f"'{cy}' no es numérica."); return
                d = self.df.head(30)
                ax.bar(d[cx].astype(str), d[cy], color=FO_GREEN, edgecolor=FO_HOVER, alpha=0.85)
                ax.set_xlabel(cx, color=FO_GREEN, fontsize=10, family=FO_FONT)
                ax.set_ylabel(cy, color=FO_GREEN, fontsize=10, family=FO_FONT)
                ax.set_title(f"{cy} por {cx} (30)", color=FO_GREEN, fontsize=12, weight="bold", family=FO_FONT)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

            elif ct == "Pastel (Pie)":
                if not pd.api.types.is_numeric_dtype(self.df[cy]):
                    messagebox.showerror("Error", f"'{cy}' no es numérica."); return
                pie = self.df.groupby(cx)[cy].sum().nlargest(8)
                greens = ["#4af626", "#39ff14", "#2dd50e", "#20ab08", "#16a308", "#0d7a05", "#085204", "#042902"]
                ax.pie(pie.values, labels=pie.index.astype(str), autopct="%1.1f%%",
                       colors=greens[:len(pie)], textprops={"color": FO_GREEN, "fontsize": 9, "family": FO_FONT})
                ax.set_title(f"{cy} por {cx} (Top 8)", color=FO_GREEN, fontsize=12, weight="bold", family=FO_FONT)

            self.fig.tight_layout()
            self.canvas_widget.draw()
            logger.info(f"Gráfico: {ct} — X={cx}, Y={cy}")
        except Exception as e:
            logger.error(f"Error gráfico: {e}"); logger.error(traceback.format_exc())
            messagebox.showerror("Error", str(e))

    # ══════════════════════════════════════════════
    # Tab 4 — ML ENGINE
    # ══════════════════════════════════════════════
    def _build_tab_ml(self):
        mf = ctk.CTkFrame(self.tab_ml, fg_color="transparent")
        mf.pack(fill="both", expand=True, padx=5, pady=5)
        mf.columnconfigure(0, weight=0)
        mf.columnconfigure(1, weight=1)
        mf.columnconfigure(2, weight=1)
        mf.rowconfigure(0, weight=1)

        # ── Panel 1: Config ──
        p1 = self._fo_card(mf, width=260)
        p1.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        p1.grid_propagate(False)

        self._fo_label(p1, "// CONFIG ML", 14).pack(pady=(12, 10))

        self._fo_label(p1, "Modelo:", 10).pack(padx=12, anchor="w")
        self.opt_ml_model = self._fo_option(p1, list(ML_MODELS.keys()), 230)
        self.opt_ml_model.pack(padx=12, pady=(2, 8))

        self._fo_label(p1, "Variable Objetivo (Y):", 10).pack(padx=12, anchor="w")
        self.opt_ml_target = self._fo_option(p1, ["-- Cargar datos --"], 230)
        self.opt_ml_target.pack(padx=12, pady=(2, 8))

        self._fo_label(p1, "Predictoras (X):", 10).pack(padx=12, anchor="w", pady=(4, 2))
        self.ml_checkbox_frame = ctk.CTkScrollableFrame(
            p1, fg_color=FO_BLACK, corner_radius=4, height=160,
            border_width=1, border_color=FO_GREEN,
        )
        self.ml_checkbox_frame.pack(fill="x", padx=12, pady=(0, 10))
        self.ml_checkboxes: dict[str, ctk.CTkCheckBox] = {}
        self.ml_checkbox_vars: dict[str, ctk.BooleanVar] = {}

        self._fo_btn(p1, "> ENTRENAR MODELO", self.entrenar_modelo,
                     width=230, height=36).pack(pady=(5, 15))

        # ── Panel 2: Inferencia ──
        p2 = self._fo_card(mf)
        p2.grid(row=0, column=1, sticky="nsew", padx=5, pady=0)

        self._fo_label(p2, "// INFERENCIA MANUAL", 14).pack(pady=(12, 8))
        ctk.CTkLabel(p2, text="Entrene un modelo para habilitar prediccion.",
                     text_color=FO_DIM, font=ctk.CTkFont(family=FO_FONT, size=10),
                     wraplength=250).pack(pady=(0, 5))

        self.frame_inputs_manuales = ctk.CTkScrollableFrame(
            p2, fg_color=FO_BLACK, corner_radius=4,
            border_width=1, border_color=FO_GREEN,
        )
        self.frame_inputs_manuales.pack(fill="both", expand=True, padx=10, pady=(0, 5))
        self.manual_entries: dict[str, ctk.CTkEntry] = {}

        self._fo_btn(p2, "> PREDECIR VALOR", self.predecir_manual,
                     width=220, height=34).pack(pady=5)

        self.lbl_prediction_result = ctk.CTkLabel(
            p2, text="---", font=ctk.CTkFont(family=FO_FONT, size=26, weight="bold"),
            text_color=FO_GREEN,
        )
        self.lbl_prediction_result.pack(pady=(5, 12))

        # ── Panel 3: Resultados ──
        p3 = self._fo_card(mf)
        p3.grid(row=0, column=2, sticky="nsew", padx=(5, 0), pady=0)
        self._fo_label(p3, "// RESULTADOS", 14).pack(pady=(12, 8))

        # Métricas
        met = ctk.CTkFrame(p3, fg_color=FO_BLACK, corner_radius=4, border_width=1, border_color=FO_GREEN)
        met.pack(fill="x", padx=10, pady=(0, 5))

        r1 = ctk.CTkFrame(met, fg_color="transparent")
        r1.pack(fill="x", padx=10, pady=(8, 2))
        self._fo_label(r1, "MSE:", 12).pack(side="left")
        self.lbl_mse = ctk.CTkLabel(r1, text="---", text_color=FO_GREEN,
                                    font=ctk.CTkFont(family=FO_FONT, size=12))
        self.lbl_mse.pack(side="left", padx=(8, 0))

        r2 = ctk.CTkFrame(met, fg_color="transparent")
        r2.pack(fill="x", padx=10, pady=(2, 8))
        self._fo_label(r2, "R²:", 12).pack(side="left")
        self.lbl_r2 = ctk.CTkLabel(r2, text="---", text_color=FO_GREEN,
                                   font=ctk.CTkFont(family=FO_FONT, size=12))
        self.lbl_r2.pack(side="left", padx=(8, 0))

        # ML Chart
        self.fig_ml = Figure(figsize=(5, 4), dpi=100, facecolor=FO_BLACK)
        ax_ml = self.fig_ml.add_subplot(111)
        ax_ml.set_facecolor(FO_BG)
        ax_ml.text(0.5, 0.5, "ENTRENE UN MODELO PARA VER\nEL GRAFICO DE PREDICCIONES",
                   ha="center", va="center", fontsize=10, color=FO_DIM,
                   transform=ax_ml.transAxes, family=FO_FONT)
        self._style_ax_fallout(ax_ml)

        self.canvas_ml = FigureCanvasTkAgg(self.fig_ml, master=p3)
        self.canvas_ml.draw()
        self.canvas_ml.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(5, 10))

    # ── ML — Actualizar Opciones ──
    def actualizar_opciones_ml(self):
        if self.df is None or self.df.empty: return
        nc = self.df.select_dtypes(include="number").columns.tolist()
        if not nc: return
        self.opt_ml_target.configure(values=nc)
        self.opt_ml_target.set("Weight" if "Weight" in nc else nc[0])

        for w in self.ml_checkbox_frame.winfo_children(): w.destroy()
        self.ml_checkboxes.clear(); self.ml_checkbox_vars.clear()

        for col in nc:
            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                self.ml_checkbox_frame, text=col, variable=var,
                font=ctk.CTkFont(family=FO_FONT, size=11), text_color=FO_GREEN,
                fg_color=FO_GREEN, hover_color=FO_HOVER,
                border_color=FO_GREEN, checkmark_color=FO_BLACK,
            )
            cb.pack(anchor="w", padx=5, pady=2)
            self.ml_checkboxes[col] = cb
            self.ml_checkbox_vars[col] = var
        logger.info(f"Opciones ML actualizadas: {len(nc)} columnas numéricas.")

    # ── ML — Entrenar ──
    def entrenar_modelo(self):
        if not self._check_df_loaded(): return
        tc = self.opt_ml_target.get()
        sx = [c for c, v in self.ml_checkbox_vars.items() if v.get()]
        mn = self.opt_ml_model.get()
        if not sx:
            messagebox.showerror("SIN PREDICTORES", "Seleccione al menos una variable X."); return
        if tc in sx:
            messagebox.showerror("CONFLICTO", f"'{tc}' no puede ser X y Y."); return
        try:
            df_c = self.df[sx + [tc]].dropna()
            if df_c.shape[0] < 10:
                messagebox.showerror("DATOS INSUFICIENTES", f"Solo {df_c.shape[0]} filas válidas (min 10)."); return
            X, y = df_c[sx].values, df_c[tc].values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            model = ML_MODELS[mn](); model.fit(X_tr, y_tr)
            y_p = model.predict(X_te)
            mse, r2 = mean_squared_error(y_te, y_p), r2_score(y_te, y_p)

            self.modelo_actual = model; self.columnas_x_entrenadas = sx
            self.lbl_mse.configure(text=f"{mse:.4f}")
            self.lbl_r2.configure(text=f"{r2:.4f}")

            # Chart
            self.fig_ml.clear()
            ax = self.fig_ml.add_subplot(111)
            ax.set_facecolor(FO_BG)
            self._style_ax_fallout(ax)
            ax.scatter(y_te, y_p, c=FO_GREEN, edgecolors=FO_HOVER, alpha=0.85, s=45, label="Predicciones")
            lims = [min(y_te.min(), y_p.min()), max(y_te.max(), y_p.max())]
            ax.plot(lims, lims, "--", color="#ff3333", linewidth=1.5, alpha=0.7, label="Ideal (y=x)")
            ax.set_xlabel("Valores Reales", color=FO_GREEN, fontsize=10, family=FO_FONT)
            ax.set_ylabel("Predicciones", color=FO_GREEN, fontsize=10, family=FO_FONT)
            ax.set_title(f"{mn} // R²={r2:.3f}", color=FO_GREEN, fontsize=12, weight="bold", family=FO_FONT)
            ax.legend(fontsize=9, loc="upper left", facecolor=FO_BLACK, edgecolor=FO_GREEN, labelcolor=FO_GREEN)
            self.fig_ml.tight_layout(); self.canvas_ml.draw()

            self._populate_manual_inputs(sx)
            logger.info(f"Modelo: {mn} | MSE={mse:.4f} | R²={r2:.4f}")
        except Exception as e:
            logger.error(f"Error: {e}"); logger.error(traceback.format_exc())
            messagebox.showerror("Error", str(e))

    def _populate_manual_inputs(self, columns: list[str]):
        for w in self.frame_inputs_manuales.winfo_children(): w.destroy()
        self.manual_entries.clear()
        for col in columns:
            row = ctk.CTkFrame(self.frame_inputs_manuales, fg_color="transparent")
            row.pack(fill="x", padx=5, pady=3)
            self._fo_label(row, col, 10, width=120, anchor="w").pack(side="left")
            entry = self._fo_entry(row, placeholder=f"Valor {col}", width=120, height=28)
            entry.pack(side="left", padx=(5, 0))
            self.manual_entries[col] = entry

    # ── ML — Predicción ──
    def predecir_manual(self):
        if self.modelo_actual is None:
            messagebox.showerror("SIN MODELO", "Primero entrene un modelo."); return
        try:
            vals = []
            for col in self.columnas_x_entrenadas:
                raw = self.manual_entries[col].get().strip()
                if not raw:
                    messagebox.showerror("CAMPO VACÍO", f"'{col}' está vacío."); return
                try:
                    vals.append(float(raw))
                except ValueError:
                    messagebox.showerror("VALOR INVÁLIDO", f"'{raw}' no es numérico para '{col}'."); return
            pred = self.modelo_actual.predict(np.array([vals]))[0]
            self.lbl_prediction_result.configure(text=f"{pred:.2f}")
            logger.info(f"Predicción: {dict(zip(self.columnas_x_entrenadas, vals))} → {pred:.2f}")
        except Exception as e:
            logger.error(f"Error: {e}"); logger.error(traceback.format_exc())
            messagebox.showerror("Error", str(e))


# ──────────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────────
def main():
    try:
        load_dotenv()
        logger.info("Iniciando Recomp Master Pro...")
        app = RecompApp()
        app.mainloop()
    except Exception as e:
        logger.critical(f"Error crítico: {e}")
        logger.critical(traceback.format_exc())


if __name__ == "__main__":
    main()
