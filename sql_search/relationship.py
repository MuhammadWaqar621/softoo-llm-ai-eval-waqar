"""
Database relationship analysis functionality.
"""
import pandas as pd
from sqlalchemy import inspect, text


class RelationshipAnalyzer:
    """Analyzes database relationships."""

    def __init__(self, db_connection):
        """
        Initialize relationship analyzer.

        Args:
            db_connection: DatabaseConnection instance
        """
        self.db_connection = db_connection

    def get_table_relationships(self):
        """
        Generate a visualization of table relationships for key tables.

        Returns:
            DataFrame: Table relationships
        """
        if not self.db_connection.is_connected():
            return "Database connection required to extract relationships."

        try:
            # Get foreign key relationships
            inspector = inspect(self.db_connection.engine)
            relationships = []

            # Focus on important schemas
            important_schemas = ["Sales", "Production", "Person", "HumanResources"]

            for schema in important_schemas:
                tables = inspector.get_table_names(schema=schema)

                for table in tables:
                    fks = inspector.get_foreign_keys(table, schema=schema)
                    if fks:
                        for fk in fks:
                            ref_schema = fk.get('referred_schema', schema)
                            relationships.append({
                                'from_table': f"{schema}.{table}",
                                'to_table': f"{ref_schema}.{fk['referred_table']}",
                                'from_column': ', '.join(fk['constrained_columns']),
                                'to_column': ', '.join(fk['referred_columns'])
                            })

            # Create a table of relationships
            if relationships:
                rel_df = pd.DataFrame(relationships)
                return rel_df
            else:
                return "No relationships found."

        except Exception as e:
            return f"Error extracting relationships: {str(e)}"

    def analyze_schema(self):
        """
        Provide analysis of the database schema.

        Returns:
            DataFrame: Schema summary
        """
        if not self.db_connection.is_connected():
            return "Database connection required for schema analysis."

        try:
            # Get schema structure
            inspector = inspect(self.db_connection.engine)
            schema_data = []

            # Get all schemas
            query = text("SELECT DISTINCT TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES ORDER BY TABLE_SCHEMA")
            schemas = [row[0] for row in self.db_connection.connection.execute(query)]

            for schema in schemas:
                tables = inspector.get_table_names(schema=schema)
                for table in tables:
                    columns = inspector.get_columns(table, schema=schema)
                    schema_data.append({
                        'schema': schema,
                        'table': table,
                        'columns': len(columns)
                    })

            # Create summary of schema
            if schema_data:
                schema_df = pd.DataFrame(schema_data)
                schema_summary = schema_df.groupby('schema').agg(
                    table_count=('table', 'count'),
                    avg_columns=('columns', 'mean')
                ).reset_index()
                return schema_summary
            else:
                return "No schema data found."

        except Exception as e:
            return f"Error analyzing schema: {str(e)}"