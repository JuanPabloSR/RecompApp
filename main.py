import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
from dotenv import load_dotenv
import pandas as pd
import os
import logging
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RecompApp")

# Constantes
EXPECTED_COLUMNS = [
    "Date", "Protein", "Fats", "Sat Fats",
    "Carbs", "Sugars", "Calories", "Weight", "Body Fat"
]
MAX_ROWS_DISPLAY = 100


class RecompApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.df: pd.DataFrame | None = None
        self._setup_window()
        self._setup_styles()
        self._build_ui()

    # ──────────────────────────────────────────────
    # Configuración de Ventana
    # ──────────────────────────────────────────────
    def _setup_window(self):
        """Configura la ventana principal."""
        self.title("Recomp Master Pro")
        self.geometry("1200x800")
        self.minsize(900, 600)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

    # ──────────────────────────────────────────────
    # Estilo ttk (para el Treeview en dark mode)
    # ──────────────────────────────────────────────
    def _setup_styles(self):
        """Configura estilos del ttk.Treeview para dark mode."""
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(
            "Dark.Treeview",
            background="#1e1e2e",
            foreground="#d4d4d4",
            fieldbackground="#1e1e2e",
            rowheight=28,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Dark.Treeview.Heading",
            background="#0f3460",
            foreground="#e0e0e0",
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Dark.Treeview",
            background=[("selected", "#264f78")],
            foreground=[("selected", "#ffffff")],
        )

    # ──────────────────────────────────────────────
    # Construcción de la UI
    # ──────────────────────────────────────────────
    def _build_ui(self):
        """Construye el layout principal con CTkTabview."""
        try:
            self.tabview = ctk.CTkTabview(self, corner_radius=10)
            self.tabview.pack(fill="both", expand=True, padx=15, pady=15)

            self.tab_datos = self.tabview.add("📊 Datos")
            self.tab_gestion = self.tabview.add("⚙️ Gestión")
            self.tab_graficos = self.tabview.add("📈 Gráficos")
            self.tab_ml = self.tabview.add("🤖 Machine Learning")

            self._build_tab_datos()
            self._build_tab_gestion()
            self._build_tab_graficos()
            self._build_tab_placeholder(self.tab_ml, "Módulo de Machine Learning (Próximamente)")

            logger.info("Interfaz inicializada correctamente.")

        except Exception as e:
            logger.error(f"Error construyendo la UI: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"No se pudo inicializar la interfaz:\n{e}")

    def _build_tab_placeholder(self, parent, message: str):
        """Genera un placeholder centrado para pestañas vacías."""
        label = ctk.CTkLabel(
            parent, text=message, text_color="#888888",
            font=ctk.CTkFont(size=16, weight="normal"),
        )
        label.place(relx=0.5, rely=0.5, anchor="center")

    # ──────────────────────────────────────────────
    # Tab 1 — Explorador de Datos
    # ──────────────────────────────────────────────
    def _build_tab_datos(self):
        """Construye la pestaña de Explorador de Datos."""
        top_frame = ctk.CTkFrame(self.tab_datos, fg_color="transparent")
        top_frame.pack(fill="x", padx=10, pady=(10, 5))

        btn_cargar = ctk.CTkButton(
            top_frame, text="📂 Cargar Dataset",
            command=self._on_load_dataset,
            width=180, height=38, corner_radius=8,
            fg_color="#0f3460", hover_color="#1a5276",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        btn_cargar.pack(side="left", padx=(0, 15))

        self.file_status_label = ctk.CTkLabel(
            top_frame, text="Ningún archivo cargado",
            text_color="#888888", font=ctk.CTkFont(size=12),
        )
        self.file_status_label.pack(side="left")

        tree_frame = ctk.CTkFrame(self.tab_datos, fg_color="transparent")
        tree_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        scroll_y = ttk.Scrollbar(tree_frame, orient="vertical")
        scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")

        self.treeview = ttk.Treeview(
            tree_frame, style="Dark.Treeview",
            yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set,
            show="headings",
        )
        scroll_y.config(command=self.treeview.yview)
        scroll_x.config(command=self.treeview.xview)

        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.treeview.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")

    def _on_load_dataset(self):
        """Abre un diálogo para seleccionar un archivo y lo carga en self.df."""
        file_path = filedialog.askopenfilename(
            title="Seleccionar dataset",
            filetypes=[
                ("Archivos CSV", "*.csv"),
                ("Archivos Excel", "*.xlsx"),
            ],
        )
        if not file_path:
            return

        file_name = os.path.basename(file_path)
        logger.info(f"Archivo seleccionado: {file_path}")

        try:
            ext = os.path.splitext(file_name)[1].lower()
            if ext == ".csv":
                self.df = pd.read_csv(file_path)
            elif ext == ".xlsx":
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Formato no soportado: {ext}")

            logger.info(f"Dataset cargado: {self.df.shape[0]} filas × {self.df.shape[1]} columnas")
            self.file_status_label.configure(
                text=f"✔ {file_name}  —  {self.df.shape[0]} filas × {self.df.shape[1]} columnas",
                text_color="#4caf50",
            )
            self.update_treeview()
            self.actualizar_dropdowns_graficos()

        except Exception as ex:
            logger.error(f"Error al cargar archivo: {ex}")
            logger.error(traceback.format_exc())
            self.df = None
            self.file_status_label.configure(
                text=f"✘ Error al cargar {file_name}", text_color="#ef5350",
            )
            messagebox.showerror("Error de carga", f"No se pudo leer el archivo:\n{ex}")

    def update_treeview(self):
        """Renderiza self.df en el Treeview (primeras MAX_ROWS_DISPLAY filas)."""
        if self.df is None or self.df.empty:
            return

        self.treeview.delete(*self.treeview.get_children())
        display_df = self.df.head(MAX_ROWS_DISPLAY)
        cols = list(display_df.columns)

        self.treeview["columns"] = cols
        for col in cols:
            self.treeview.heading(col, text=col, anchor="center")
            self.treeview.column(col, anchor="center", width=110, minwidth=80)

        for _, row in display_df.iterrows():
            values = [str(v) for v in row]
            self.treeview.insert("", "end", values=values)

        if self.df.shape[0] > MAX_ROWS_DISPLAY:
            self.file_status_label.configure(
                text=(
                    f"✔ {self.file_status_label.cget('text').split('—')[0].strip()}  "
                    f"—  Mostrando {MAX_ROWS_DISPLAY} de {self.df.shape[0]} filas"
                ),
                text_color="#ff9800",
            )

    # ──────────────────────────────────────────────
    # Tab 2 — Gestión y Smart Cleaning
    # ──────────────────────────────────────────────
    def _build_tab_gestion(self):
        """Construye la pestaña de Gestión con dos columnas."""

        # Contenedor principal de dos columnas
        main_frame = ctk.CTkFrame(self.tab_gestion, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # ═══════════════════════════════════════════
        # PANEL IZQUIERDO — Control
        # ═══════════════════════════════════════════
        left_panel = ctk.CTkFrame(main_frame, corner_radius=10)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        ctk.CTkLabel(
            left_panel, text="🛠️ Panel de Control",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(pady=(12, 8))

        # ── Sección: Smart Cleaning ──
        clean_frame = ctk.CTkFrame(left_panel, fg_color="#1e1e2e", corner_radius=8)
        clean_frame.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(
            clean_frame, text="Smart Cleaning",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#64b5f6",
        ).pack(pady=(10, 6))

        ctk.CTkButton(
            clean_frame, text="🧹 Eliminar Duplicados",
            command=self._remove_duplicates,
            width=220, height=34, corner_radius=8,
            fg_color="#0f3460", hover_color="#1a5276",
            font=ctk.CTkFont(size=12),
        ).pack(pady=4)

        ctk.CTkButton(
            clean_frame, text="🩹 Imputar Nulos (Media)",
            command=self._impute_nulls,
            width=220, height=34, corner_radius=8,
            fg_color="#0f3460", hover_color="#1a5276",
            font=ctk.CTkFont(size=12),
        ).pack(pady=(4, 12))

        # ── Sección: CRUD — Agregar Registro ──
        crud_frame = ctk.CTkFrame(left_panel, fg_color="#1e1e2e", corner_radius=8)
        crud_frame.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(
            crud_frame, text="➕ Agregar Entrenamiento",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#64b5f6",
        ).pack(pady=(10, 6))

        # Campos del formulario
        fields_data = [
            ("Date (DD.MM.YY)", "entry_date"),
            ("Calories", "entry_calories"),
            ("Protein", "entry_protein"),
            ("Weight", "entry_weight"),
        ]
        self.crud_entries: dict[str, ctk.CTkEntry] = {}

        for label_text, attr_name in fields_data:
            row_frame = ctk.CTkFrame(crud_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=15, pady=2)

            ctk.CTkLabel(
                row_frame, text=label_text, width=130, anchor="w",
                font=ctk.CTkFont(size=11), text_color="#aaaaaa",
            ).pack(side="left")

            entry = ctk.CTkEntry(
                row_frame, width=150, height=30, corner_radius=6,
                fg_color="#2a2a3e", border_color="#3a3a5e",
                placeholder_text=label_text.split("(")[0].strip(),
            )
            entry.pack(side="left", padx=(5, 0))
            self.crud_entries[attr_name] = entry

        ctk.CTkButton(
            crud_frame, text="✅ Añadir Entrenamiento",
            command=self._add_record,
            width=220, height=34, corner_radius=8,
            fg_color="#2e7d32", hover_color="#388e3c",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(pady=(8, 14))

        # ═══════════════════════════════════════════
        # PANEL DERECHO — Consola de Registro
        # ═══════════════════════════════════════════
        right_panel = ctk.CTkFrame(main_frame, corner_radius=10)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        ctk.CTkLabel(
            right_panel, text="📋 Consola de Registro",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(pady=(12, 8))

        self.console_log = ctk.CTkTextbox(
            right_panel, corner_radius=8,
            fg_color="#1a1a2e", text_color="#c8c8dc",
            font=ctk.CTkFont(family="Consolas", size=12),
            state="disabled",
        )
        self.console_log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Mensaje de bienvenida en la consola
        self._log_to_console("Consola lista. Cargue un dataset para comenzar.")

    # ──────────────────────────────────────────────
    # Helpers de Gestión
    # ──────────────────────────────────────────────
    def _log_to_console(self, message: str):
        """Imprime un mensaje con timestamp en la consola de registro."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_log.configure(state="normal")
        self.console_log.insert("end", f"[{timestamp}]  {message}\n")
        self.console_log.see("end")
        self.console_log.configure(state="disabled")

    def _check_df_loaded(self) -> bool:
        """Verifica que el DataFrame esté cargado. Muestra error si no."""
        if self.df is None or self.df.empty:
            messagebox.showerror(
                "Sin datos",
                "No hay un dataset cargado.\nVe a la pestaña 📊 Datos y carga un archivo primero.",
            )
            return False
        return True

    # ──────────────────────────────────────────────
    # Smart Cleaning
    # ──────────────────────────────────────────────
    def _remove_duplicates(self):
        """Elimina filas duplicadas del DataFrame."""
        if not self._check_df_loaded():
            return

        try:
            before = self.df.shape[0]
            self.df.drop_duplicates(inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            removed = before - self.df.shape[0]

            msg = (
                f"Se eliminaron {removed} fila(s) duplicada(s)."
                if removed > 0
                else "No se encontraron duplicados."
            )
            self._log_to_console(msg)
            logger.info(msg)
            self.update_treeview()

        except Exception as e:
            logger.error(f"Error eliminando duplicados: {e}")
            self._log_to_console(f"ERROR: {e}")
            messagebox.showerror("Error", f"No se pudieron eliminar duplicados:\n{e}")

    def _impute_nulls(self):
        """Imputa valores nulos con la media de cada columna numérica."""
        if not self._check_df_loaded():
            return

        try:
            numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
            nulls_before = self.df[numeric_cols].isnull().sum().sum()

            if nulls_before == 0:
                self._log_to_console("No se encontraron valores nulos en columnas numéricas.")
                return

            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            nulls_after = self.df[numeric_cols].isnull().sum().sum()

            msg = (
                f"Se imputaron {nulls_before} valor(es) nulo(s) con la media.\n"
                f"  Columnas afectadas: {', '.join(numeric_cols)}"
            )
            self._log_to_console(msg)
            logger.info(msg)
            self.update_treeview()

        except Exception as e:
            logger.error(f"Error imputando nulos: {e}")
            self._log_to_console(f"ERROR: {e}")
            messagebox.showerror("Error", f"No se pudieron imputar los nulos:\n{e}")

    # ──────────────────────────────────────────────
    # CRUD — Agregar Registro
    # ──────────────────────────────────────────────
    def _add_record(self):
        """Toma los valores del formulario CRUD y añade una nueva fila a self.df."""
        if not self._check_df_loaded():
            return

        try:
            date_val = self.crud_entries["entry_date"].get().strip()
            cal_val = self.crud_entries["entry_calories"].get().strip()
            prot_val = self.crud_entries["entry_protein"].get().strip()
            weight_val = self.crud_entries["entry_weight"].get().strip()

            # Validar que no estén vacíos
            if not all([date_val, cal_val, prot_val, weight_val]):
                messagebox.showerror("Campos vacíos", "Todos los campos son obligatorios.")
                return

            # Validar numéricos
            try:
                cal_num = float(cal_val)
                prot_num = float(prot_val)
                weight_num = float(weight_val)
            except ValueError:
                messagebox.showerror(
                    "Valor inválido",
                    "Calories, Protein y Weight deben ser valores numéricos.",
                )
                return

            # Construir nueva fila con las columnas del DataFrame
            new_row = {col: None for col in self.df.columns}
            new_row["Date"] = date_val
            new_row["Calories"] = cal_num
            new_row["Protein"] = prot_num
            new_row["Weight"] = weight_num

            new_df = pd.DataFrame([new_row])
            self.df = pd.concat([self.df, new_df], ignore_index=True)

            # Limpiar campos
            for entry in self.crud_entries.values():
                entry.delete(0, "end")

            msg = f"Registro añadido: Date={date_val}, Cal={cal_num}, Prot={prot_num}, Weight={weight_num}"
            self._log_to_console(msg)
            logger.info(msg)
            self.update_treeview()

        except Exception as e:
            logger.error(f"Error añadiendo registro: {e}")
            self._log_to_console(f"ERROR: {e}")
            messagebox.showerror("Error", f"No se pudo añadir el registro:\n{e}")
    # ──────────────────────────────────────────────
    # Tab 3 — Gráficos con Matplotlib
    # ──────────────────────────────────────────────
    def _build_tab_graficos(self):
        """Construye la pestaña de Gráficos con dos paneles."""

        main_frame = ctk.CTkFrame(self.tab_graficos, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        main_frame.columnconfigure(0, weight=0)   # panel controles (fijo)
        main_frame.columnconfigure(1, weight=1)    # panel canvas (expandible)
        main_frame.rowconfigure(0, weight=1)

        # ═══════════ PANEL IZQUIERDO — Controles ═══════════
        ctrl_panel = ctk.CTkFrame(main_frame, width=240, corner_radius=10)
        ctrl_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        ctrl_panel.grid_propagate(False)

        ctk.CTkLabel(
            ctrl_panel, text="🎛️ Controles",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(pady=(12, 10))

        # Variable X
        ctk.CTkLabel(
            ctrl_panel, text="Variable X", text_color="#aaaaaa",
            font=ctk.CTkFont(size=11),
        ).pack(padx=15, anchor="w")
        self.opt_var_x = ctk.CTkOptionMenu(
            ctrl_panel, values=["-- Cargar datos --"],
            width=210, fg_color="#2a2a3e", button_color="#0f3460",
            button_hover_color="#1a5276",
        )
        self.opt_var_x.pack(padx=15, pady=(2, 8))

        # Variable Y
        ctk.CTkLabel(
            ctrl_panel, text="Variable Y", text_color="#aaaaaa",
            font=ctk.CTkFont(size=11),
        ).pack(padx=15, anchor="w")
        self.opt_var_y = ctk.CTkOptionMenu(
            ctrl_panel, values=["-- Cargar datos --"],
            width=210, fg_color="#2a2a3e", button_color="#0f3460",
            button_hover_color="#1a5276",
        )
        self.opt_var_y.pack(padx=15, pady=(2, 8))

        # Tipo de gráfico
        ctk.CTkLabel(
            ctrl_panel, text="Tipo de Gráfico", text_color="#aaaaaa",
            font=ctk.CTkFont(size=11),
        ).pack(padx=15, anchor="w")
        self.opt_chart_type = ctk.CTkOptionMenu(
            ctrl_panel,
            values=["Dispersión (Scatter)", "Barras (Bar)", "Pastel (Pie)"],
            width=210, fg_color="#2a2a3e", button_color="#0f3460",
            button_hover_color="#1a5276",
        )
        self.opt_chart_type.pack(padx=15, pady=(2, 15))

        # Botón generar
        ctk.CTkButton(
            ctrl_panel, text="📊 Generar Gráfico",
            command=self._generate_chart,
            width=210, height=38, corner_radius=8,
            fg_color="#2e7d32", hover_color="#388e3c",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(pady=(0, 15))

        # ═══════════ PANEL DERECHO — Canvas ═══════════
        canvas_panel = ctk.CTkFrame(main_frame, corner_radius=10)
        canvas_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        ctk.CTkLabel(
            canvas_panel, text="📈 Visualización",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(pady=(12, 5))

        # Matplotlib Figure + Canvas
        plt.style.use("dark_background")
        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor="#2b2b3c")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e2e")
        self.ax.text(
            0.5, 0.5, "Seleccione variables y genere un gráfico",
            ha="center", va="center", fontsize=13, color="#888888",
            transform=self.ax.transAxes,
        )

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=canvas_panel)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def actualizar_dropdowns_graficos(self):
        """Actualiza los OptionMenus X e Y con las columnas del DataFrame."""
        if self.df is None or self.df.empty:
            return
        cols = list(self.df.columns)
        self.opt_var_x.configure(values=cols)
        self.opt_var_y.configure(values=cols)
        # Seleccionar defaults razonables
        if len(cols) >= 2:
            self.opt_var_x.set(cols[0])
            self.opt_var_y.set(cols[1])
        elif len(cols) == 1:
            self.opt_var_x.set(cols[0])
            self.opt_var_y.set(cols[0])

    def _generate_chart(self):
        """Genera el gráfico según las selecciones del usuario."""
        if not self._check_df_loaded():
            return

        col_x = self.opt_var_x.get()
        col_y = self.opt_var_y.get()
        chart_type = self.opt_chart_type.get()

        # Validar que las columnas existan
        if col_x not in self.df.columns:
            messagebox.showerror("Error", f"La columna '{col_x}' no existe en el dataset.")
            return

        try:
            # Limpiar figura
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_facecolor("#1e1e2e")

            if chart_type == "Dispersión (Scatter)":
                # Validar numéricas
                if not pd.api.types.is_numeric_dtype(self.df[col_x]):
                    messagebox.showerror("Error", f"'{col_x}' no es numérica. Seleccione una columna numérica.")
                    return
                if col_y not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col_y]):
                    messagebox.showerror("Error", f"'{col_y}' no es numérica. Seleccione una columna numérica.")
                    return

                ax.scatter(
                    self.df[col_x], self.df[col_y],
                    c="#42a5f5", edgecolors="#1565c0", alpha=0.8, s=50,
                )
                ax.set_xlabel(col_x, color="#cccccc", fontsize=11)
                ax.set_ylabel(col_y, color="#cccccc", fontsize=11)
                ax.set_title(f"{col_y} vs {col_x}", color="#e0e0e0", fontsize=13, weight="bold")

            elif chart_type == "Barras (Bar)":
                if not pd.api.types.is_numeric_dtype(self.df[col_y]):
                    messagebox.showerror("Error", f"'{col_y}' no es numérica. Seleccione una columna numérica para Y.")
                    return

                data = self.df.head(30)  # Limitar a 30 barras para legibilidad
                bars = ax.bar(
                    data[col_x].astype(str), data[col_y],
                    color="#42a5f5", edgecolor="#1565c0", alpha=0.85,
                )
                ax.set_xlabel(col_x, color="#cccccc", fontsize=11)
                ax.set_ylabel(col_y, color="#cccccc", fontsize=11)
                ax.set_title(f"{col_y} por {col_x} (primeros 30)", color="#e0e0e0", fontsize=13, weight="bold")
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

            elif chart_type == "Pastel (Pie)":
                # Para Pie solo se usa la variable Y (numérica) agrupada por X
                if not pd.api.types.is_numeric_dtype(self.df[col_y]):
                    messagebox.showerror("Error", f"'{col_y}' no es numérica. Seleccione una columna numérica para Y.")
                    return

                pie_data = self.df.groupby(col_x)[col_y].sum().nlargest(8)
                colors_pie = ["#42a5f5", "#66bb6a", "#ffa726", "#ef5350",
                              "#ab47bc", "#26c6da", "#8d6e63", "#78909c"]
                ax.pie(
                    pie_data.values,
                    labels=pie_data.index.astype(str),
                    autopct="%1.1f%%",
                    colors=colors_pie[:len(pie_data)],
                    textprops={"color": "#e0e0e0", "fontsize": 9},
                )
                ax.set_title(f"{col_y} por {col_x} (Top 8)", color="#e0e0e0", fontsize=13, weight="bold")

            self.fig.tight_layout()
            self.canvas_widget.draw()
            logger.info(f"Gráfico generado: {chart_type} — X={col_x}, Y={col_y}")

        except Exception as e:
            logger.error(f"Error generando gráfico: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"No se pudo generar el gráfico:\n{e}")


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
