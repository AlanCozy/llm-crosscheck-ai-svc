"""
Prompt engine for template management and rendering.

This module provides a comprehensive prompt engine that handles loading,
caching, and rendering of Jinja2 templates for LLM prompts.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from jinja2 import Environment, FileSystemLoader, Template, TemplateError, TemplateNotFound

from ..models.llm import PromptTemplate, TemplateContext
from .logging import get_logger

logger = get_logger(__name__)


class PromptEngine:
    """
    Comprehensive prompt engine for template management.
    
    Handles loading, caching, validation, and rendering of Jinja2 templates
    for LLM prompts with support for template inheritance and custom filters.
    """
    
    def __init__(
        self,
        template_dirs: List[str],
        auto_reload: bool = False,
        cache_size: int = 100,
    ):
        """
        Initialise the prompt engine.
        
        Args:
            template_dirs: List of directories to search for templates
            auto_reload: Whether to automatically reload templates when changed
            cache_size: Maximum number of templates to cache
        """
        self.template_dirs = [Path(dir_path) for dir_path in template_dirs]
        self.auto_reload = auto_reload
        self.cache_size = cache_size
        
        # Template cache
        self._template_cache: Dict[str, PromptTemplate] = {}
        self._jinja_cache: Dict[str, Template] = {}
        
        # Validate template directories
        self._validate_template_dirs()
        
        # Set up Jinja2 environment
        self._setup_jinja_environment()
        
        logger.info(
            "Initialised prompt engine",
            extra={
                "template_dirs": [str(d) for d in self.template_dirs],
                "auto_reload": auto_reload,
                "cache_size": cache_size,
            }
        )
    
    def load_template(self, template_name: str) -> PromptTemplate:
        """
        Load a template by name.
        
        Args:
            template_name: Name of the template to load
            
        Returns:
            Loaded prompt template
            
        Raises:
            TemplateNotFound: If template cannot be found
            TemplateError: If template is invalid
        """
        # Check cache first
        if template_name in self._template_cache and not self.auto_reload:
            return self._template_cache[template_name]
        
        # Find template file
        template_path = self._find_template_file(template_name)
        if not template_path:
            raise TemplateNotFound(f"Template '{template_name}' not found")
        
        # Load and parse template
        template = self._load_template_file(template_path, template_name)
        
        # Cache the template
        if len(self._template_cache) >= self.cache_size:
            self._evict_oldest_template()
        
        self._template_cache[template_name] = template
        
        logger.debug(
            f"Loaded template '{template_name}'",
            extra={
                "template_name": template_name,
                "template_path": str(template_path),
                "category": template.category,
                "version": template.version,
            }
        )
        
        return template
    
    def render_template(
        self,
        template_name: str,
        context: TemplateContext,
        validate_variables: bool = True,
    ) -> str:
        """
        Render a template with the given context.
        
        Args:
            template_name: Name of the template to render
            context: Template context with variables
            validate_variables: Whether to validate required variables
            
        Returns:
            Rendered template content
            
        Raises:
            TemplateNotFound: If template cannot be found
            TemplateError: If rendering fails
            ValueError: If required variables are missing
        """
        # Load the template
        template = self.load_template(template_name)
        
        # Validate required variables
        if validate_variables:
            self._validate_template_variables(template, context)
        
        # Get or create Jinja2 template
        jinja_template = self._get_jinja_template(template_name, template.template)
        
        try:
            # Render the template
            rendered = jinja_template.render(**context.variables)
            
            logger.debug(
                f"Rendered template '{template_name}'",
                extra={
                    "template_name": template_name,
                    "context_variables": list(context.variables.keys()),
                    "rendered_length": len(rendered),
                }
            )
            
            return rendered
            
        except TemplateError as e:
            logger.error(
                f"Failed to render template '{template_name}'",
                extra={
                    "template_name": template_name,
                    "error": str(e),
                    "context_variables": list(context.variables.keys()),
                }
            )
            raise TemplateError(f"Failed to render template '{template_name}': {str(e)}")
    
    def render_template_string(
        self,
        template_string: str,
        context: TemplateContext,
    ) -> str:
        """
        Render a template string directly.
        
        Args:
            template_string: Template content as string
            context: Template context with variables
            
        Returns:
            Rendered template content
            
        Raises:
            TemplateError: If rendering fails
        """
        try:
            template = self.jinja_env.from_string(template_string)
            return template.render(**context.variables)
        except TemplateError as e:
            logger.error(
                f"Failed to render template string",
                extra={
                    "error": str(e),
                    "template_length": len(template_string),
                    "context_variables": list(context.variables.keys()),
                }
            )
            raise TemplateError(f"Failed to render template string: {str(e)}")
    
    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of template names
        """
        templates = []
        
        for template_dir in self.template_dirs:
            if not template_dir.exists():
                continue
            
            for template_file in template_dir.rglob("*.j2"):
                # Calculate relative path as template name
                template_name = str(template_file.relative_to(template_dir)).replace(".j2", "")
                
                # Filter by category if specified
                if category:
                    template_category = template_file.parent.name
                    if template_category != category:
                        continue
                
                templates.append(template_name)
        
        return sorted(templates)
    
    def validate_template(self, template_name: str) -> bool:
        """
        Validate a template's syntax.
        
        Args:
            template_name: Name of the template to validate
            
        Returns:
            True if template is valid, False otherwise
        """
        try:
            template = self.load_template(template_name)
            self._get_jinja_template(template_name, template.template)
            return True
        except (TemplateNotFound, TemplateError):
            return False
    
    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear()
        self._jinja_cache.clear()
        logger.info("Cleared template cache")
    
    def _validate_template_dirs(self) -> None:
        """Validate that template directories exist."""
        for template_dir in self.template_dirs:
            if not template_dir.exists():
                logger.warning(
                    f"Template directory does not exist: {template_dir}",
                    extra={"template_dir": str(template_dir)}
                )
    
    def _setup_jinja_environment(self) -> None:
        """Set up the Jinja2 environment."""
        # Convert Path objects to strings for FileSystemLoader
        template_dir_strings = [str(d) for d in self.template_dirs if d.exists()]
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir_strings),
            auto_reload=self.auto_reload,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Add custom filters
        self._add_custom_filters()
    
    def _add_custom_filters(self) -> None:
        """Add custom Jinja2 filters for LLM prompts."""
        
        def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
            """Truncate text to maximum length."""
            if len(text) <= max_length:
                return text
            return text[:max_length - len(suffix)] + suffix
        
        def format_list(items: List[Any], separator: str = ", ", last_separator: str = " and ") -> str:
            """Format a list with proper separators."""
            if not items:
                return ""
            if len(items) == 1:
                return str(items[0])
            if len(items) == 2:
                return f"{items[0]}{last_separator}{items[1]}"
            return separator.join(str(item) for item in items[:-1]) + last_separator + str(items[-1])
        
        def quote_text(text: str, quote_char: str = '"') -> str:
            """Quote text with specified character."""
            return f"{quote_char}{text}{quote_char}"
        
        # Register filters
        self.jinja_env.filters["truncate_text"] = truncate_text
        self.jinja_env.filters["format_list"] = format_list
        self.jinja_env.filters["quote"] = quote_text
    
    def _find_template_file(self, template_name: str) -> Optional[Path]:
        """Find template file by name."""
        template_filename = f"{template_name}.j2"
        
        for template_dir in self.template_dirs:
            if not template_dir.exists():
                continue
            
            # Try direct path
            template_path = template_dir / template_filename
            if template_path.exists():
                return template_path
            
            # Try nested paths
            for template_file in template_dir.rglob(template_filename):
                relative_path = str(template_file.relative_to(template_dir))
                if relative_path.replace(".j2", "") == template_name:
                    return template_file
        
        return None
    
    def _load_template_file(self, template_path: Path, template_name: str) -> PromptTemplate:
        """Load template from file."""
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract metadata from template comments if present
            metadata = self._extract_template_metadata(content)
            
            # Determine category from path
            category = template_path.parent.name if template_path.parent.name != "prompts" else None
            
            return PromptTemplate(
                name=template_name,
                description=metadata.get("description"),
                template=content,
                version=metadata.get("version", "1.0.0"),
                category=category,
                tags=metadata.get("tags", []),
                required_variables=metadata.get("required_variables", []),
                optional_variables=metadata.get("optional_variables", []),
            )
            
        except Exception as e:
            raise TemplateError(f"Failed to load template file '{template_path}': {str(e)}")
    
    def _extract_template_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from template comments."""
        metadata = {}
        lines = content.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line.startswith("{#") or not line.endswith("#}"):
                continue
            
            # Remove comment markers
            comment = line[2:-2].strip()
            
            # Parse metadata
            if ":" in comment:
                key, value = comment.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key in ["tags", "required_variables", "optional_variables"]:
                    # Parse comma-separated lists
                    metadata[key] = [item.strip() for item in value.split(",")]
                else:
                    metadata[key] = value
        
        return metadata
    
    def _get_jinja_template(self, template_name: str, template_content: str) -> Template:
        """Get or create Jinja2 template."""
        if template_name in self._jinja_cache and not self.auto_reload:
            return self._jinja_cache[template_name]
        
        try:
            template = self.jinja_env.from_string(template_content)
            self._jinja_cache[template_name] = template
            return template
        except TemplateError as e:
            raise TemplateError(f"Invalid template syntax in '{template_name}': {str(e)}")
    
    def _validate_template_variables(self, template: PromptTemplate, context: TemplateContext) -> None:
        """Validate that required variables are present in context."""
        missing_variables = []
        
        for required_var in template.required_variables:
            if required_var not in context.variables:
                missing_variables.append(required_var)
        
        if missing_variables:
            raise ValueError(
                f"Missing required variables for template '{template.name}': {missing_variables}"
            )
    
    def _evict_oldest_template(self) -> None:
        """Evict the oldest template from cache."""
        if self._template_cache:
            oldest_key = next(iter(self._template_cache))
            del self._template_cache[oldest_key]
            if oldest_key in self._jinja_cache:
                del self._jinja_cache[oldest_key]
