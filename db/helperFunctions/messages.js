// Imports
const { client } = require("../index");

// Create Message
async function createMessage({ conversationId, role, content }) {
    try {
        const result = await client.query(`
            INSERT INTO messages (conversation_id, role, content)
            VALUES ($1, $2, $3)
            RETURNING *;
        `, [conversationId, role, content]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating message:', error);
        throw error;
    }
};

// Get All Messages
async function getAllMessages() {
    try {
        const result = await client.query('SELECT * FROM messages;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all messages:', error);
        throw error;
    }
};

// Get Message By ID
async function getMessageById(id) {
    try {
        const result = await client.query('SELECT * FROM messages WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting message by ID:', error);
        throw error;
    }
};

// Get Messages By Conversation ID
async function getMessagesByConversationId(conversationId) {
    try {
        const result = await client.query('SELECT * FROM messages WHERE conversation_id = $1 ORDER BY created_at;', [conversationId]);
        return result.rows;
    } catch (error) {
        console.error('Error getting messages by conversation ID:', error);
        throw error;
    }
};

// Update Message
async function updateMessage(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    try {
        const result = await client.query(`
            UPDATE messages
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating message:', error);
        throw error;
    }
};

// Delete Message
async function deleteMessage(id) {
    try {
        const result = await client.query('DELETE FROM messages WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting message:', error);
        throw error;
    }
};

module.exports = {
    createMessage,
    getAllMessages,
    getMessageById,
    getMessagesByConversationId,
    updateMessage,
    deleteMessage
};