"""
Schema extraction functionality for the SQL RAG System.
"""
import pandas as pd
from sqlalchemy import inspect, text
from rich.console import Console

console = Console()


class SchemaExtractor:
    """Extracts and processes database schema information."""

    def __init__(self, db_connection):
        """
        Initialize schema extractor.

        Args:
            db_connection: DatabaseConnection instance
        """
        self.db_connection = db_connection

    def extract_schema(self):
        """
        Extract schema information from the database.

        Returns:
            str: Formatted schema information
        """
        if not self.db_connection.is_connected():
            return self._get_static_schema_description()

        try:
            inspector = inspect(self.db_connection.engine)
            schema_info = "AdventureWorks Database Schema:\n\n"

            # Get all schemas
            query = text("SELECT DISTINCT TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES ORDER BY TABLE_SCHEMA")
            schemas = [row[0] for row in self.db_connection.connection.execute(query)]

            for schema in schemas:
                # Get all tables for this schema
                tables = inspector.get_table_names(schema=schema)

                for table_name in tables:
                    full_table_name = f"{schema}.{table_name}"
                    schema_info += f"Table: {full_table_name}\n"

                    # Get table description if available (from sys.extended_properties)
                    desc_query = text(
                        f"""
                        SELECT ep.value 
                        FROM sys.extended_properties ep
                        JOIN sys.tables t ON ep.major_id = t.object_id
                        JOIN sys.schemas s ON t.schema_id = s.schema_id
                        WHERE ep.name = 'MS_Description'
                        AND ep.minor_id = 0
                        AND s.name = '{schema}'
                        AND t.name = '{table_name}'
                        """
                    )

                    description_result = self.db_connection.connection.execute(desc_query).fetchone()
                    table_description = description_result[0] if description_result else "No description available"
                    schema_info += f"Description: {table_description}\n"

                    # Get columns and their info
                    schema_info += "Columns:\n"
                    columns = inspector.get_columns(table_name, schema=schema)

                    for column in columns:
                        col_name = column['name']
                        col_type = str(column['type'])

                        # Get column description if available
                        col_desc_query = text(
                            f"""
                            SELECT ep.value 
                            FROM sys.extended_properties ep
                            JOIN sys.tables t ON ep.major_id = t.object_id
                            JOIN sys.columns c ON ep.major_id = c.object_id AND ep.minor_id = c.column_id
                            JOIN sys.schemas s ON t.schema_id = s.schema_id
                            WHERE ep.name = 'MS_Description'
                            AND s.name = '{schema}'
                            AND t.name = '{table_name}'
                            AND c.name = '{col_name}'
                            """
                        )

                        col_desc_result = self.db_connection.connection.execute(col_desc_query).fetchone()
                        col_description = col_desc_result[0] if col_desc_result else "No description available"

                        schema_info += f"  - {col_name} ({col_type}): {col_description}\n"

                    # Get primary key information
                    pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
                    if pk_constraint and 'constrained_columns' in pk_constraint:
                        pk_cols = ', '.join(pk_constraint['constrained_columns'])
                        schema_info += f"Primary Key: {pk_cols}\n"

                    # Get foreign key information
                    fks = inspector.get_foreign_keys(table_name, schema=schema)
                    if fks:
                        schema_info += "Foreign Keys:\n"
                        for fk in fks:
                            referred_schema = fk.get('referred_schema')
                            referred_table = fk.get('referred_table')
                            referred_columns = fk.get('referred_columns')
                            constrained_columns = fk.get('constrained_columns')

                            fk_info = f"  - {', '.join(constrained_columns)} -> "
                            if referred_schema:
                                fk_info += f"{referred_schema}."
                            fk_info += f"{referred_table}({', '.join(referred_columns)})"
                            schema_info += fk_info + "\n"

                    schema_info += "\n"

            # Limit the schema information size if it's too large
            if len(schema_info) > 50000:  # Reasonable size to include in prompts
                console.print("[yellow]Schema is very large, including only the most important tables[/yellow]")
                # Include only the main sales-related tables if schema is too large
                important_schemas = ["Sales", "Production", "Person", "HumanResources"]
                important_tables = [
                    "SalesOrderHeader", "SalesOrderDetail", "Product", "Customer",
                    "Person", "ProductCategory", "SalesTerritory"
                ]

                filtered_schema = "AdventureWorks Database Schema (Important Tables Only):\n\n"

                for line in schema_info.split("\n"):
                    if line.startswith("Table:"):
                        table_name = line.replace("Table:", "").strip()
                        schema_name = table_name.split('.')[0] if '.' in table_name else ""
                        base_table_name = table_name.split('.')[-1] if '.' in table_name else table_name

                        if (schema_name in important_schemas or
                                any(imp_table in base_table_name for imp_table in important_tables)):
                            include_table = True
                        else:
                            include_table = False

                    if include_table:
                        filtered_schema += line + "\n"

                return filtered_schema

            return schema_info
        except Exception as e:
            console.print(f"[red]Error extracting schema from database: {e}[/red]")
            return self._get_static_schema_description()

    def _get_static_schema_description(self):
        """
        Provide a static description of the AdventureWorks schema.
        This is a fallback if database connection is unavailable.

        Returns:
            str: Static schema description
        """
        schema = {
            # Sales schema tables
            "sales_order_header": {
                "table_name": "Sales.SalesOrderHeader",
                "description": "Contains sales order header data including customer ID, order date, and total amount due.",
                "columns": [
                    {"name": "SalesOrderID", "type": "int", "description": "Primary key for SalesOrderHeader records."},
                    {"name": "RevisionNumber", "type": "tinyint", "description": "Incremental number to track changes to the sales order."},
                    {"name": "OrderDate", "type": "datetime", "description": "Date the order was placed."},
                    {"name": "DueDate", "type": "datetime", "description": "Date the order is due to the customer."},
                    {"name": "ShipDate", "type": "datetime", "description": "Date the order was shipped to the customer."},
                    {"name": "Status", "type": "tinyint",
                     "description": "Order current status: 1 = In process, 2 = Approved, 3 = Backordered, 4 = Rejected, 5 = Shipped, 6 = Cancelled"},
                    {"name": "OnlineOrderFlag", "type": "bit", "description": "0 = Order placed by sales person, 1 = Order placed online."},
                    {"name": "SalesOrderNumber", "type": "nvarchar", "description": "Unique sales order identification number."},
                    {"name": "PurchaseOrderNumber", "type": "nvarchar", "description": "Customer purchase order number reference."},
                    {"name": "CustomerID", "type": "int", "description": "Foreign key to Customer.CustomerID."},
                    {"name": "TerritoryID", "type": "int", "description": "Territory in which the sale was made."},
                    {"name": "SubTotal", "type": "money", "description": "Sales subtotal. Computed as SUM(SalesOrderDetail.LineTotal)."},
                    {"name": "TaxAmt", "type": "money", "description": "Tax amount."},
                    {"name": "Freight", "type": "money", "description": "Shipping cost."},
                    {"name": "TotalDue", "type": "money", "description": "Total due from customer. Computed as Subtotal + TaxAmt + Freight."}
                ]
            },
            # Additional tables would be defined here
            # ...
        }

        # Format schema for LLM consumption
        formatted_schema = "AdventureWorks Database Schema:\n\n"
        for table_id, table_info in schema.items():
            formatted_schema += f"Table: {table_info['table_name']}\n"
            formatted_schema += f"Description: {table_info['description']}\n"
            formatted_schema += "Columns:\n"
            for column in table_info['columns']:
                formatted_schema += f"  - {column['name']} ({column['type']}): {column['description']}\n"
            formatted_schema += "\n"

        return formatted_schema