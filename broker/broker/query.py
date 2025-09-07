"""
Pandas-style query engine for GPU search and filtering
"""

from typing import Any, List, Union

from .types import GPUOffer


class QueryField:
    """Represents a queryable field on GPU offers"""
    
    def __init__(self, field_name: str):
        self.field_name = field_name
    
    def contains(self, value: str) -> 'QueryCondition':
        """Field contains substring"""
        return QueryCondition(self.field_name, 'contains', value)
    
    def startswith(self, value: str) -> 'QueryCondition':
        """Field starts with string"""
        return QueryCondition(self.field_name, 'startswith', value)
    
    def endswith(self, value: str) -> 'QueryCondition':
        """Field ends with string"""
        return QueryCondition(self.field_name, 'endswith', value)
    
    def isin(self, values: List[Any]) -> 'QueryCondition':
        """Field value is in list"""
        return QueryCondition(self.field_name, 'isin', values)
    
    def __eq__(self, value: Any) -> 'QueryCondition':
        """Field equals value"""
        return QueryCondition(self.field_name, 'eq', value)
    
    def __ne__(self, value: Any) -> 'QueryCondition':
        """Field not equals value"""
        return QueryCondition(self.field_name, 'ne', value)
    
    def __lt__(self, value: Any) -> 'QueryCondition':
        """Field less than value"""
        return QueryCondition(self.field_name, 'lt', value)
    
    def __le__(self, value: Any) -> 'QueryCondition':
        """Field less than or equal to value"""
        return QueryCondition(self.field_name, 'le', value)
    
    def __gt__(self, value: Any) -> 'QueryCondition':
        """Field greater than value"""
        return QueryCondition(self.field_name, 'gt', value)
    
    def __ge__(self, value: Any) -> 'QueryCondition':
        """Field greater than or equal to value"""
        return QueryCondition(self.field_name, 'ge', value)


class QueryCondition:
    """Single condition in a query"""
    
    def __init__(self, field: str, operator: str, value: Any):
        self.field = field
        self.operator = operator
        self.value = value
    
    def __and__(self, other: Union['QueryCondition', 'QueryExpression']) -> 'QueryExpression':
        """Combine with AND"""
        return QueryExpression('and', [self, other])
    
    def __or__(self, other: Union['QueryCondition', 'QueryExpression']) -> 'QueryExpression':
        """Combine with OR"""
        return QueryExpression('or', [self, other])
    
    def evaluate(self, obj: GPUOffer) -> bool:
        """Evaluate condition against GPU offer"""
        field_value = getattr(obj, self.field, None)
        
        if field_value is None:
            return False
            
        if self.operator == 'contains':
            return self.value.lower() in str(field_value).lower()
        elif self.operator == 'startswith':
            return str(field_value).lower().startswith(self.value.lower())
        elif self.operator == 'endswith':
            return str(field_value).lower().endswith(self.value.lower())
        elif self.operator == 'isin':
            return field_value in self.value
        elif self.operator == 'eq':
            return field_value == self.value
        elif self.operator == 'ne':
            return field_value != self.value
        elif self.operator == 'lt':
            return field_value < self.value
        elif self.operator == 'le':
            return field_value <= self.value
        elif self.operator == 'gt':
            return field_value > self.value
        elif self.operator == 'ge':
            return field_value >= self.value
        
        return False
    
    def __repr__(self) -> str:
        return f"QueryCondition({self.field} {self.operator} {self.value})"


class QueryExpression:
    """Boolean combination of conditions"""
    
    def __init__(self, operator: str, conditions: List[Union[QueryCondition, 'QueryExpression']]):
        self.operator = operator
        self.conditions = conditions
    
    def __and__(self, other: Union[QueryCondition, 'QueryExpression']) -> 'QueryExpression':
        """Combine with AND"""
        return QueryExpression('and', [self, other])
    
    def __or__(self, other: Union[QueryCondition, 'QueryExpression']) -> 'QueryExpression':
        """Combine with OR"""
        return QueryExpression('or', [self, other])
    
    def evaluate(self, obj: GPUOffer) -> bool:
        """Evaluate expression against GPU offer"""
        if self.operator == 'and':
            return all(cond.evaluate(obj) for cond in self.conditions)
        elif self.operator == 'or':
            return any(cond.evaluate(obj) for cond in self.conditions)
        return False
    
    def __repr__(self) -> str:
        return f"QueryExpression({self.operator}: {self.conditions})"


class GPUQuery:
    """Pandas-style query interface for GPU offers"""
    
    # Query fields
    gpu_type = QueryField('gpu_type')
    price_per_hour = QueryField('price_per_hour')
    vram_gb = QueryField('vram_gb')          # GPU VRAM in GB
    memory_gb = QueryField('memory_gb')      # Alias for vram_gb (backward compatibility)
    provider = QueryField('provider')
    availability = QueryField('availability')
    region = QueryField('region')
    vcpu = QueryField('vcpu')
    storage_gb = QueryField('storage_gb')
    spot = QueryField('spot')
    cuda_version = QueryField('cuda_version')
    driver_version = QueryField('driver_version')
    cloud_type = QueryField('cloud_type')
    manufacturer = QueryField('manufacturer')


# Type alias for query types
QueryType = Union[QueryCondition, QueryExpression]