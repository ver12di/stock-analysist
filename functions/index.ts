export async function handleRequest(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const info = {
    message: "Cloudflare Worker is running for stock-analysist",
    path: url.pathname,
    note: "Update this handler to proxy Streamlit or provide an API once deployed.",
  };
  return Response.json(info, { status: 200 });
}

export default {
  async fetch(request: Request): Promise<Response> {
    return handleRequest(request);
  },
};
