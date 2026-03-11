import flet as ft
from dotenv import load_dotenv
import os
import logging
import traceback

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RecompApp")

class RecompApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.setup_page()
        self.init_ui()

    def setup_page(self):
        """Configura las propiedades estéticas y de la ventana."""
        try:
            self.page.title = "Recomp Master Pro"
            self.page.theme_mode = ft.ThemeMode.DARK
            self.page.padding = 20
            
            # Estética minimalista
            self.page.bgcolor = ft.Colors.BACKGROUND
            
            # Configuración de ventana (Enfoque en Escritorio)
            self.page.window_width = 1200
            self.page.window_height = 800
            self.page.window_min_width = 900
            self.page.window_min_height = 600
        except Exception as e:
            logger.error(f"Error en setup_page: {e}")
            logger.error(traceback.format_exc())

    def init_ui(self):
        """Inicializa los componentes principales de la interfaz."""
        try:
            # Tab 1: Datos
            tab_datos = ft.Tab(
                text="Datos",
                icon=ft.Icons.DATA_USAGE,
                content=ft.Container(
                    content=ft.Text("Módulo de Datos (Próximamente)", size=16, color=ft.Colors.WHITE54, weight=ft.FontWeight.W_300),
                    alignment=ft.alignment.center
                )
            )

            # Tab 2: Gestión
            tab_gestion = ft.Tab(
                text="Gestión",
                icon=ft.Icons.SETTINGS_APPLICATIONS,
                content=ft.Container(
                    content=ft.Text("Módulo de Gestión (Próximamente)", size=16, color=ft.Colors.WHITE54, weight=ft.FontWeight.W_300),
                    alignment=ft.alignment.center
                )
            )

            # Tab 3: Gráficos
            tab_graficos = ft.Tab(
                text="Gráficos",
                icon=ft.Icons.PIE_CHART,
                content=ft.Container(
                    content=ft.Text("Módulo de Gráficos (Próximamente)", size=16, color=ft.Colors.WHITE54, weight=ft.FontWeight.W_300),
                    alignment=ft.alignment.center
                )
            )

            # Tab 4: Machine Learning
            tab_ml = ft.Tab(
                text="Machine Learning",
                icon=ft.Icons.MODEL_TRAINING,
                content=ft.Container(
                    content=ft.Text("Módulo de Machine Learning (Próximamente)", size=16, color=ft.Colors.WHITE54, weight=ft.FontWeight.W_300),
                    alignment=ft.alignment.center
                )
            )

            # Contenedor principal de pestañas
            self.tabs = ft.Tabs(
                selected_index=0,
                animation_duration=300,
                tabs=[tab_datos, tab_gestion, tab_graficos, tab_ml],
                expand=1
            )

            # Añadir las pestañas a la pantalla principal
            self.page.add(self.tabs)
            logger.info("Interfaz inicializada correctamente.")

        except Exception as e:
            logger.error(f"Error inicializando UI: {e}")
            logger.error(traceback.format_exc())
            self.show_error("Ha ocurrido un error al cargar la interfaz principal.")

    def show_error(self, message: str):
        """Método auxiliar para mostrar mensajes de error al usuario."""
        try:
            self.page.snack_bar = ft.SnackBar(
                content=ft.Text(message, color=ft.Colors.WHITE),
                bgcolor=ft.Colors.ERROR,
                action="OK"
            )
            self.page.snack_bar.open = True
            self.page.update()
        except Exception as e:
            logger.error(f"Fallo al mostrar el snackbar de error: {e}")

def main(page: ft.Page):
    try:
        # Cargar variables de entorno
        load_dotenv()
        logger.info("Iniciando Recomp Master Pro...")
        
        # Instanciar la aplicación principal
        app = RecompApp(page)
    except Exception as e:
        logger.critical(f"Error crítico durante el inicio: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    try:
        # Punto de entrada de la aplicación Flet
        ft.run(main)
    except Exception as e:
        logger.critical(f"Fallo al lanzar la aplicación Flet: {e}")
