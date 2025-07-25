if FORMAT == "latex" then
  return
end

local i = 1
local tbl = {}
local fig = {}
local kAriaExpanded = "aria-expanded"

-- Default words
local figureword = "Figure"
local tableword = "Table"
local refhyperlinks = true

local function gettablefig(m)
  if m.language then
    if m.language["crossref-fig-title"] then
      figureword = pandoc.utils.stringify(m.language["crossref-fig-title"])
    end
    if m.language["crossref-tbl-title"] then
      tableword = pandoc.utils.stringify(m.language["crossref-tbl-title"])
    end
  end
  if m["ref-hyperlink"] == false then
    refhyperlinks = false
  end
end

local function figtblconvert(ct)
  -- Build map of floats
  while quarto._quarto.ast.custom_node_data[tostring(i)] do
    local float = quarto._quarto.ast.custom_node_data[tostring(i)]
    if float and float.identifier and float.attributes then
      if string.find(float.identifier, "^tbl%-") and not tbl[float.identifier] then
        tbl[float.identifier] = float.attributes.prefix .. float.attributes.tblnum
      elseif string.find(float.identifier, "^fig%-") and not fig[float.identifier] then
        fig[float.identifier] = float.attributes.prefix .. float.attributes.fignum
      end
    end
    i = i + 1
  end

  if #ct.citations ~= 1 then
    return nil
  end

  local id = ct.citations[1].id
  local label = nil
  local word = nil

  if string.find(id, "^tbl%-") then
    label = tbl[id]
    word = tableword
  elseif string.find(id, "^fig%-") then
    label = fig[id]
    word = figureword
  end

  if not label or not word then
    return nil -- gracefully skip if label is not found
  end

  local floatreftext = pandoc.Inlines({
    pandoc.Str(word),
    pandoc.Str('\u{a0}'),
    pandoc.Str(label)
  })

  if refhyperlinks then
    local reflink = pandoc.Link(floatreftext, "#" .. id)
    reflink.classes = { "quarto-xref" }
    reflink.attributes[kAriaExpanded] = "false"
    return reflink
  else
    return floatreftext
  end
end

return {
  { Meta = gettablefig },
  { Cite = figtblconvert }
}
