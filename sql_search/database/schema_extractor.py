"""
Schema extraction functionality for the SQL RAG System.
"""
from sqlalchemy import inspect, text
from rich.console import Console

console = Console()


class SchemaExtractor:
    """Extracts and processes database schema information for AdventureWorksLT."""

    def __init__(self, db_connection):
        """
        Initialize schema extractor.

        Args:
            db_connection: DatabaseConnection instance
        """
        self.db_connection = db_connection

    def extract_schema(self):
        """
        Extract schema information from the AdventureWorksLT database.

        Returns:
            str: Formatted schema information
        """
        if not self.db_connection.is_connected():
            return self._get_static_schema_description()

        try:
            inspector = inspect(self.db_connection.engine)
            schema_info = "AdventureWorksLT Database Schema:\n\n"

            # Get all schemas (focusing on SalesLT)
            target_schemas = ["SalesLT"]
            
            for schema in target_schemas:
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

            return schema_info
        except Exception as e:
            console.print(f"[red]Error extracting schema from database: {e}[/red]")
            return self._get_static_schema_description()

    def _get_static_schema_description(self):
        """
        Provide a static description of the AdventureWorksLT schema.
        This is a fallback if database connection is unavailable.

        Returns:
            str: Static schema description
        """
        schema = {
            # SalesLT schema tables
            "customer": {
                "table_name": "SalesLT.Customer",
                "description": "Contains customers who purchase products.",
                "columns": [
                    {"name": "CustomerID", "type": "int", "description": "Primary key for Customer records."},
                    {"name": "NameStyle", "type": "bit", "description": "0 = Western style names, 1 = Eastern style names."},
                    {"name": "Title", "type": "nvarchar", "description": "Title such as Mr., Ms., or Dr."},
                    {"name": "FirstName", "type": "nvarchar", "description": "First name of the customer."},
                    {"name": "MiddleName", "type": "nvarchar", "description": "Middle name of the customer."},
                    {"name": "LastName", "type": "nvarchar", "description": "Last name of the customer."},
                    {"name": "CompanyName", "type": "nvarchar", "description": "Customer's company name."},
                    {"name": "EmailAddress", "type": "nvarchar", "description": "E-mail address for the customer."},
                    {"name": "Phone", "type": "nvarchar", "description": "Phone number associated with the customer."},
                    {"name": "PasswordHash", "type": "varchar", "description": "Password for the customer account."},
                    {"name": "PasswordSalt", "type": "varchar", "description": "Random value concatenated with the password for additional security."}
                ]
            },
            "sales_order_header": {
                "table_name": "SalesLT.SalesOrderHeader",
                "description": "Contains sales order header data.",
                "columns": [
                    {"name": "SalesOrderID", "type": "int", "description": "Primary key for SalesOrderHeader records."},
                    {"name": "RevisionNumber", "type": "tinyint", "description": "Incremental number to track changes to the sales order."},
                    {"name": "OrderDate", "type": "datetime", "description": "Date the order was placed."},
                    {"name": "DueDate", "type": "datetime", "description": "Date the order is due to the customer."},
                    {"name": "ShipDate", "type": "datetime", "description": "Date the order was shipped to the customer."},
                    {"name": "Status", "type": "tinyint", "description": "Order current status. 1 = In process; 2 = Approved; 3 = Backordered; 4 = Rejected; 5 = Shipped; 6 = Cancelled"},
                    {"name": "OnlineOrderFlag", "type": "bit", "description": "0 = Order placed by sales person. 1 = Order placed online by customer."},
                    {"name": "SalesOrderNumber", "type": "nvarchar", "description": "Unique sales order identification number."},
                    {"name": "PurchaseOrderNumber", "type": "nvarchar", "description": "Customer purchase order number reference."},
                    {"name": "CustomerID", "type": "int", "description": "Foreign key to Customer.CustomerID."},
                    {"name": "ShipToAddressID", "type": "int", "description": "The ID of the location to send goods."},
                    {"name": "BillToAddressID", "type": "int", "description": "The ID of the location to send the invoice."},
                    {"name": "ShipMethod", "type": "nvarchar", "description": "Shipping method. Foreign key to ShipMethod.ShipMethodID."},
                    {"name": "SubTotal", "type": "money", "description": "Sales subtotal. Computed as SUM(SalesOrderDetail.LineTotal)."},
                    {"name": "TaxAmt", "type": "money", "description": "Tax amount."},
                    {"name": "Freight", "type": "money", "description": "Shipping cost."},
                    {"name": "TotalDue", "type": "money", "description": "Total due from customer. Computed as Subtotal + TaxAmt + Freight."}
                ]
            },
            "sales_order_detail": {
                "table_name": "SalesLT.SalesOrderDetail",
                "description": "Contains individual line items associated with sales orders.",
                "columns": [
                    {"name": "SalesOrderID", "type": "int", "description": "Primary key. Foreign key to SalesOrderHeader.SalesOrderID."},
                    {"name": "SalesOrderDetailID", "type": "int", "description": "Primary key. One increment per product sold."},
                    {"name": "OrderQty", "type": "smallint", "description": "Quantity ordered per product."},
                    {"name": "ProductID", "type": "int", "description": "Foreign key to Product.ProductID."},
                    {"name": "UnitPrice", "type": "money", "description": "Selling price of a single product."},
                    {"name": "UnitPriceDiscount", "type": "money", "description": "Discount amount."},
                    {"name": "LineTotal", "type": "numeric", "description": "Per product subtotal. Computed as UnitPrice * (1 - UnitPriceDiscount) * OrderQty."}
                ]
            },
            "product": {
                "table_name": "SalesLT.Product",
                "description": "Products sold or used in the manufacturing of sold products.",
                "columns": [
                    {"name": "ProductID", "type": "int", "description": "Primary key for Product records."},
                    {"name": "Name", "type": "nvarchar", "description": "Name of the product."},
                    {"name": "ProductNumber", "type": "nvarchar", "description": "Unique product identification number."},
                    {"name": "Color", "type": "nvarchar", "description": "Product color."},
                    {"name": "StandardCost", "type": "money", "description": "Standard cost of the product."},
                    {"name": "ListPrice", "type": "money", "description": "Selling price."},
                    {"name": "Size", "type": "nvarchar", "description": "Product size."},
                    {"name": "Weight", "type": "decimal", "description": "Product weight."},
                    {"name": "ProductCategoryID", "type": "int", "description": "Foreign key to ProductCategory.ProductCategoryID."},
                    {"name": "ProductModelID", "type": "int", "description": "Foreign key to ProductModel.ProductModelID."},
                    {"name": "SellStartDate", "type": "datetime", "description": "Date the product was available for sale."},
                    {"name": "SellEndDate", "type": "datetime", "description": "Date the product was no longer available for sale."}
                ]
            },
            "product_category": {
                "table_name": "SalesLT.ProductCategory",
                "description": "Categories for products.",
                "columns": [
                    {"name": "ProductCategoryID", "type": "int", "description": "Primary key for ProductCategory records."},
                    {"name": "ParentProductCategoryID", "type": "int", "description": "Foreign key to ProductCategory.ProductCategoryID."},
                    {"name": "Name", "type": "nvarchar", "description": "Category name."}
                ]
            },
            "address": {
                "table_name": "SalesLT.Address",
                "description": "Street address information.",
                "columns": [
                    {"name": "AddressID", "type": "int", "description": "Primary key for Address records."},
                    {"name": "AddressLine1", "type": "nvarchar", "description": "First street address line."},
                    {"name": "AddressLine2", "type": "nvarchar", "description": "Second street address line."},
                    {"name": "City", "type": "nvarchar", "description": "City name."},
                    {"name": "StateProvince", "type": "nvarchar", "description": "State or province."},
                    {"name": "CountryRegion", "type": "nvarchar", "description": "Country or region."},
                    {"name": "PostalCode", "type": "nvarchar", "description": "Postal code."}
                ]
            },
            "customer_address": {
                "table_name": "SalesLT.CustomerAddress",
                "description": "Cross-reference table mapping customers to their addresses.",
                "columns": [
                    {"name": "CustomerID", "type": "int", "description": "Primary key. Foreign key to Customer.CustomerID."},
                    {"name": "AddressID", "type": "int", "description": "Primary key. Foreign key to Address.AddressID."},
                    {"name": "AddressType", "type": "nvarchar", "description": "The type of address."}
                ]
            }
        }

        # Format schema for LLM consumption
        formatted_schema = "AdventureWorksLT Database Schema:\n\n"
        for table_id, table_info in schema.items():
            formatted_schema += f"Table: {table_info['table_name']}\n"
            formatted_schema += f"Description: {table_info['description']}\n"
            formatted_schema += "Columns:\n"
            for column in table_info['columns']:
                formatted_schema += f"  - {column['name']} ({column['type']}): {column['description']}\n"
            formatted_schema += "\n"

        return formatted_schema