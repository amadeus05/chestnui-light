"""Легаси-точка входа `import etl`: полный реэкспорт `src.data.etl` без изменения логики."""
from src.data.etl import *  # noqa: F403

if __name__ == "__main__":
    main()
